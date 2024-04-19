# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
# Changes
# - Removed all references to tensorflow
# - adapt to lightning training framework
# - https://arxiv.org/pdf/1905.13077.pdf paper

"""Hierarchical Probabilistic U-Net."""

import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.uq_methods import BaseModule

from ..models.hierarchical_prob_unet import (
    LagrangeMultiplier,
    MovingAverage,
    _HierarchicalCore,
    _StitchingDecoder,
)
from .utils import (
    default_segmentation_metrics,
    process_segmentation_prediction,
    save_image_predictions,
)


class HierarchicalProbUNet(BaseModule):
    """Hierarchical Probabilistic U-Net.

    If you use this model, please cite the following paper:

    - https://arxiv.org/pdf/1905.13077.pdf
    """

    valid_loss_types = ["elbo", "geco"]
    valid_tasks = ["multiclass", "binary"]
    pred_dir_name = "preds"

    def __init__(
        self,
        latent_dims: tuple[int, ...] = (1, 1, 1, 1),
        channels_per_block: Optional[tuple[int, ...]] = None,
        num_in_channels: int = 3,
        num_classes: int = 2,
        down_channels_per_block: Optional[tuple[int, ...]] = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
        loss_type: str = "geco",
        kappa: int = 10,
        beta: Optional[float] = 100,
        top_k_percentage: Optional[float] = None,
        deterministic_top_k: Optional[int] = None,
        num_samples: int = 5,
        task: str = "multiclass",
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new HierarchicalProbUNET.

        Args:
            latent_dims: A tuple of integers specifying the dimensions of the latents
                at each scale. The length of the tuple indicates the number of U-Net
                decoder scales that have latents
            channels_per_block: A tuple of integers specifying the number of output
                channels for each block or None. If None, the default values are used
            num_in_channels: the number of input channels
            num_classes: An integer specifying the number of output classes.
            down_channels_per_block: A tuple of integers specifying the number of
                intermediate channels for each block or None. If None, the
                intermediate channels are chosen equal to channels_per_block
            activation_fn: A callable activation function
            convs_per_block: An integer specifying the number of convolutional layers
            blocks_per_level: An integer specifying the number of residual blocks per
                level
            model: the hierarchical probabilistic u-net model
            loss_type: the type of loss to use, either "elbo" or "geco"
            kappa: the kappa parameter for the geco loss
            beta: the beta parameter for the elbo loss
            top_k_percentage: the percentage of top loss_mask pixels to use for the
                loss
            deterministic_top_k: An optional percentage for the top-k loss_mask.
            num_samples: the number of samples to use during prediction
            task: task type, either "multiclass" or "binary"
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.num_in_channels = num_in_channels
        self.down_channels_per_block = down_channels_per_block
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level

        self._build_model()

        assert (
            loss_type in self.valid_loss_types
        ), f"Loss type {loss_type} not valid, please choose from {self.valid_loss_types}"  # noqa: E501
        self.loss_type = loss_type

        assert (
            task in self.valid_tasks
        ), f"Task {task} not valid, please choose from {self.valid_tasks}."
        self.task = task

        # TODO check that beta can only be used with elbo loss
        self.beta = beta
        # TODO check that kappa can only be used with geco loss
        self.kappa = kappa
        self.top_k_percentage = top_k_percentage
        self.deterministic_top_k = deterministic_top_k

        self.num_samples = num_samples

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._cache = ()

        if self.loss_type == "geco":
            self._lagmul = LagrangeMultiplier()
            self._moving_average = MovingAverage()

        self.setup_task()

    def _build_model(self) -> None:
        """Build the HierarchicalProbUnet model."""
        # these are default arguments from the original code
        # TODO need to add input channels
        base_channels = 24
        default_channels_per_block = (
            self.num_in_channels + self.latent_dims[0],
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            8 * base_channels,
        )
        if self.channels_per_block is None:
            self.channels_per_block = default_channels_per_block
        if self.down_channels_per_block is None:
            self.all_gatherdown_channels_per_block = tuple(
                [i / 2 for i in default_channels_per_block]
            )

        self._prior = _HierarchicalCore(
            latent_dims=self.latent_dims,
            channels_per_block=self.channels_per_block,
            down_channels_per_block=self.down_channels_per_block,
            activation_fn=self.activation_fn,
            convs_per_block=self.convs_per_block,
            blocks_per_level=self.blocks_per_level,
        )

        self._posterior = _HierarchicalCore(
            latent_dims=self.latent_dims,
            channels_per_block=self.channels_per_block,
            down_channels_per_block=self.down_channels_per_block,
            activation_fn=self.activation_fn,
            convs_per_block=self.convs_per_block,
            blocks_per_level=self.blocks_per_level,
        )

        self._f_comb = _StitchingDecoder(
            latent_dims=self.latent_dims,
            channels_per_block=self.channels_per_block,
            num_classes=self.num_classes,
            down_channels_per_block=self.down_channels_per_block,
            activation_fn=self.activation_fn,
            convs_per_block=self.convs_per_block,
            blocks_per_level=self.blocks_per_level,
        )

    def setup_task(self) -> None:
        """Set up the task."""
        self.train_metrics = default_segmentation_metrics(
            prefix="train", num_classes=self.num_classes, task=self.task
        )
        self.val_metrics = default_segmentation_metrics(
            prefix="val", num_classes=self.num_classes, task=self.task
        )
        self.test_metrics = default_segmentation_metrics(
            prefix="test", num_classes=self.num_classes, task=self.task
        )

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            X: the input data

        Returns:
            the output of the model
        """
        return self.model(X)

    def construct_latent_space(self, img: Tensor, seg_mask: Tensor) -> None:
        """Construct the latent space.

        Args:
            img: the input image
            seg_mask: the segmentation mask
        """
        inputs = (seg_mask, img)
        if (
            len(self._cache) == 2
            and torch.all(self._cache[0] == seg_mask)
            and torch.all(self._cache[1] == img)
        ):
            return
        else:
            self._q_sample = self._posterior(
                torch.cat([seg_mask, img], dim=1), mean=False
            )
            self._q_sample_mean = self._posterior(
                torch.cat([seg_mask, img], dim=1), mean=True
            )
            self._p_sample = self._prior(img, mean=False, z_q=None)
            self._p_sample_z_q = self._prior(img, z_q=self._q_sample["used_latents"])
            self._p_sample_z_q_mean = self._prior(
                img, z_q=self._q_sample_mean["used_latents"]
            )
            self._cache = inputs
            return

    def compute_loss(
        self, batch: dict[str, Tensor], loss_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the loss from the output of the model.

        Args:
            batch: the batch of data containing image and segmentation mask
            loss_mask: Optional tensor that masks some pixels from being
                included in the loss

        Returns:
            the loss
        """
        img = batch[self.input_key]
        seg_mask = batch[self.target_key]

        if len(seg_mask.shape) == 3:
            seg_mask_target = seg_mask.long()
            seg_mask_target = F.one_hot(seg_mask_target, num_classes=self.num_classes)
            seg_mask_target = seg_mask_target.permute(
                0, 3, 1, 2
            ).float()  # move class dim to the channel dim

            # channel dimension for concatenation
            seg_mask = seg_mask.unsqueeze(1)

        # forward pass through model that computes prior and posterior
        # but does not return anything
        self.construct_latent_space(img, seg_mask)

        # Compute reconstruction loss
        reconstruction = self.reconstruct(mean=False)
        rec_loss = ce_loss(
            reconstruction,
            seg_mask_target,
            loss_mask,
            self.top_k_percentage,
            self.deterministic_top_k,
        )

        # Compute KL divergence loss
        kl_dict = self.kl()
        kl_sum = torch.sum(torch.stack([kl for _, kl in kl_dict.items()], dim=-1))

        # TODO summaries should be logged
        # summaries["rec_loss_mean"] = rec_loss["mean"]
        # summaries["rec_loss_sum"] = rec_loss["sum"]
        # summaries["kl_sum"] = kl_sum
        # for level, kl in kl_dict.items():
        # summaries[f"kl_{level}"] = kl

        if self.loss_type == "elbo":
            loss = rec_loss["sum"] + self.beta * kl_sum
            # summaries["elbo_loss"] = loss

        elif self.loss_type == "geco":
            # TODO need to keep a record of the moving average
            ma_rec_loss = self._moving_average(rec_loss["sum"])
            mask_sum_per_instance = torch.sum(rec_loss["mask"], dim=-1)
            num_valid_pixels = torch.mean(mask_sum_per_instance)
            reconstruction_threshold = self.kappa * num_valid_pixels

            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss = lagmul * rec_constraint + kl_sum

            # TODO should be logged
            # summaries["geco_loss"] = loss
            # summaries["ma_rec_loss_mean"] = ma_rec_loss / num_valid_pixels
            # summaries["num_valid_pixels"] = num_valid_pixels
            # summaries["lagmul"] = lagmul
        return {
            "loss": loss,
            "kl_loss": kl_sum,
            "rec_loss": rec_loss["sum"],
            "reconstruction": reconstruction,
        }

    def reconstruct(self, mean: bool = False) -> dict[str, Any]:
        """Reconstruct the input.

        Args:
            mean (bool, optional): Whether to use the mean. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary containing encoder and decoder features.
        """
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self._f_comb(
            encoder_features=encoder_features, decoder_features=decoder_features
        )

    def kl(self) -> dict[int, torch.Tensor]:
        """Compute the KL divergence.

        Returns:
            A dictionary containing the KL divergence for each level
        """
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out["distributions"]
        p_dists = prior_out["distributions"]

        kl = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = torch.distributions.kl_divergence(q, p)
            kl_per_instance = torch.sum(kl_per_pixel, dim=[1, 2])
            kl[level] = torch.mean(kl_per_instance)
        return kl

    def sample(
        self, img: torch.Tensor, mean: bool = False, z_q: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        """Sample from the model.

        Args:
            img: The image tensor.
            mean: Whether to use the mean
            z_q: Latent tensor

        Returns:
            A dictionary containing encoder and decoder features
        """
        prior_out = self._prior(img, mean, z_q)
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self._f_comb(
            encoder_features=encoder_features, decoder_features=decoder_features
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        loss_dict = self.compute_loss(batch)

        self.log("train_loss", loss_dict["loss"])
        self.log("train_kl_loss", loss_dict["kl_loss"])
        self.log("train_rec_loss", loss_dict["rec_loss"])

        # compute metrics with reconstruction
        self.train_metrics(
            loss_dict["reconstruction"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        return loss_dict["loss"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        loss_dict = self.compute_loss(batch)

        self.log("val_loss", loss_dict["loss"])
        self.log("val_kl_loss", loss_dict["kl_loss"])
        self.log("val_rec_loss", loss_dict["rec_loss"])

        # compute metrics with reconstruction
        self.val_metrics(
            loss_dict["reconstruction"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        return loss_dict["loss"]

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the test loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            test prediction dict
        """
        preds = self.predict_step(batch[self.input_key])

        # compute metrics with sampled reconstruction
        self.test_metrics(
            preds["logits"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        preds = self.add_aux_data_to_dict(preds, batch)

        preds[self.target_key] = batch[self.target_key]

        return preds

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_image_predictions(outputs, batch_idx, self.pred_dir)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the prediction.

        Args:
            X: the input data
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dict
        """
        samples = torch.stack(
            [self.sample(X) for _ in range(self.num_samples)], dim=-1
        )  # shape: (batch_size, num_classes, height, width, num_samples)

        return process_segmentation_prediction(samples)

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


def _sample_gumbel(shape: torch.Size, eps: float = 1e-20) -> Tensor:
    """Transforms a uniform random variable to be standard Gumbel distributed.

    Args:
        shape: The shape of the output tensor.
        eps: A small constant for numerical stability.

    Returns:
        A tensor of the specified shape sampled from the Gumbel distribution.
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def _topk_mask(score: Tensor, k: int) -> Tensor:
    """Create a loss_mask for the top-k elements in a score tensor.

    Args:
        score: The score tensor.
        k: The number of top elements to loss_mask.

    Returns:
        A binary loss_mask tensor of the same shape as the score tensor.
    """
    _, indices = torch.topk(score, k=k)
    loss_mask = torch.zeros_like(score)
    loss_mask.scatter_(1, indices, 1)
    return loss_mask


def ce_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Optional[Tensor] = None,
    top_k_percentage: Optional[float] = None,
    deterministic: bool = False,
) -> dict[str, Tensor]:
    """Compute the cross-entropy loss between logits and labels.

    Args:
        logits: The logits tensor of shape (batch_size, num_classes, height, width)
        labels: The labels tensor of shape (batch_size, num_classes, height, width)
        mask: An optional mask tensor
        top_k_percentage: An optional percentage for the top-k loss_mask
        deterministic: A boolean indicating whether to use deterministic top-k masking

    Returns:
        A dict containing the mean and sum of the cross-entropy loss,
        and the mask
    """
    num_classes = logits.shape[1]
    y_flat = logits.view(-1, num_classes)
    t_flat = torch.argmax(labels, dim=1).view(-1)
    if mask is None:
        mask = torch.ones(y_flat.shape[0])
    else:
        mask = mask.view(-1)

    n_pixels_in_batch = y_flat.shape[0]
    xe = nn.CrossEntropyLoss(reduction="none")(y_flat, t_flat)

    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0
        k_pixels = int(n_pixels_in_batch * top_k_percentage)

        stopgrad_xe = xe.detach()
        norm_xe = stopgrad_xe / torch.sum(stopgrad_xe)

        if deterministic:
            score = torch.log(norm_xe)
        else:
            score = torch.log(norm_xe) + _sample_gumbel(norm_xe.shape)

        score = score + torch.log(mask)
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    xe = xe.view(logits.shape[0], -1)
    mask = mask.view(logits.shape[0], -1)
    ce_sum_per_instance = torch.sum(mask * xe, dim=1)
    ce_sum = torch.mean(ce_sum_per_instance)
    ce_mean = torch.sum(mask * xe) / torch.sum(mask)

    return {"mean": ce_mean, "sum": ce_sum, "mask": mask}
