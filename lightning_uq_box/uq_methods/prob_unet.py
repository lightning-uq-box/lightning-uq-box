# Copyright 2019 Stefan Knegt
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
# Changes from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/probabilistic_unet.py:
# - adapt ProbUnet implementation to lightning training framework
# - make Unet flexible to be any segmentation model

"""Probabilistic U-Net."""

import os
from typing import Any, ClassVar

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.distributions import kl

from lightning_uq_box.uq_methods import BaseModule

from ..models.prob_unet import AxisAlignedConvGaussian, Fcomb
from .utils import (
    default_segmentation_metrics,
    process_segmentation_prediction,
    save_image_predictions,
)


class ProbUNet(BaseModule):
    """Probabilistic U-Net.

    If you use this code, please cite the following paper:

    * https://arxiv.org/abs/1806.05034
    """

    valid_tasks: ClassVar[list[str]] = ["multiclass", "binary"]
    valid_fcomb_inputs: ClassVar[list[str]] = ["decoder_features", "logits"]

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        latent_dim: int = 6,
        num_filters: list[int] | None = None,
        num_convs_per_block: int = 3,
        num_convs_fcomb: int = 4,
        fcomb_filter_size: int = 32,
        beta: float = 1.0,
        fcomb_input: str = "decoder_features",
        log_sigma_bound: float = 4.0,
        num_samples: int = 5,
        task: str = "multiclass",
        criterion: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of ProbUNet.

        Args:
            model: Unet model
            latent_dim: latent dimension
            num_filters: number of filters per block in AxisAlignedConvGaussian
            num_convs_per_block: num of convs per block in AxisAlignedConvGaussian
            num_convs_fcomb: number of convolutions in fcomb
            fcomb_filter_size: filter size for the fcomb network
            beta: weight on the KL term in
                ``loss = rec_loss_sum + beta * kl_loss``, where
                ``rec_loss_sum`` is the per-image pixel sum. This matches the
                scale used by Kohl et al. (2018), so ``beta`` means the same
                thing here as in the paper and ``1.0`` is the paper default.
                It is deliberately weighted against the pixel sum rather than
                the per-pixel mean: against the mean, the equivalent value
                would be ``beta * H * W``, and any value of order 1 collapses
                the posterior onto the prior, leaving the latent carrying no
                information at all.
            fcomb_input: which feature map the latent is combined with, either
                ``"decoder_features"`` (default, as in the reference: the
                decoder output before the segmentation head, which is wider and
                not yet reduced to class scores) or ``"logits"`` (the model's
                final output). ``"logits"`` squeezes the latent through a
                ``num_classes``-wide bottleneck and is kept only for models
                that expose no decoder.
            log_sigma_bound: smooth bound on the latent log standard
                deviations in the prior and posterior encoders. Deviates from
                Kohl et al. (2018), whose unbounded exponential can diverge
                when the latent is pushed hard.
            num_samples: number of latent samples to use during prediction
            task: task type, either "multiclass" or "binary"
            criterion: reconstruction criterion, Cross Entropy Loss by default.
                Its reduction does not affect the objective: the loss is always
                built from the per-image pixel sum.
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        if num_filters is None:
            num_filters = [32, 64, 128, 192]
        super().__init__()
        self.latent_dim = latent_dim
        self.num_convs_fcomb = num_convs_fcomb
        self.beta = beta
        self.num_filters = num_filters
        self.fcomb_filter_size = fcomb_filter_size
        self.num_convs_per_block = num_convs_per_block
        self.num_samples = num_samples
        self.log_sigma_bound = log_sigma_bound

        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}."
        self.task = task

        assert fcomb_input in self.valid_fcomb_inputs, (
            f"fcomb_input must be one of {self.valid_fcomb_inputs}."
        )
        self.fcomb_input = fcomb_input

        self.model = model
        self.num_input_channels = self.num_input_features
        self.num_classes = self.num_outputs
        self.prior = AxisAlignedConvGaussian(
            self.num_input_channels,
            self.num_filters,
            self.num_convs_per_block,
            self.latent_dim,
            posterior=False,
            log_sigma_bound=log_sigma_bound,
        )
        self.posterior = AxisAlignedConvGaussian(
            self.num_input_channels,
            self.num_filters,
            self.num_convs_per_block,
            self.latent_dim,
            posterior=True,
            log_sigma_bound=log_sigma_bound,
        )
        self.fcomb_input_channels = self.num_feature_channels + self.latent_dim
        self.fcomb = Fcomb(
            self.fcomb_input_channels,
            self.fcomb_filter_size,
            self.num_classes,
            self.num_convs_fcomb,
        )

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_preds = save_preds

        self.setup_task()

    @property
    def num_feature_channels(self) -> int:
        """Channel count of the feature map the latent is combined with."""
        if self.fcomb_input == "logits":
            return self.num_classes
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            raise ValueError(
                "fcomb_input='decoder_features' requires the model to expose a "
                "`decoder` attribute, as segmentation_models_pytorch models do. "
                "Pass fcomb_input='logits' for models without one, accepting "
                "that the latent is then squeezed through a num_classes-wide "
                "bottleneck."
            )
        for module in reversed(list(decoder.modules())):
            channels = getattr(module, "out_channels", None)
            if channels is not None:
                return int(channels)
        raise ValueError(
            "could not infer the decoder's output channels; pass "
            "fcomb_input='logits' instead."
        )

    def extract_features(self, img: Tensor) -> Tensor:
        """Return the feature map that the latent is concatenated onto.

        The reference combines the latent with the decoder output *before* the
        segmentation head, where the representation is still wide. Tapping the
        final logits instead squeezes it through a ``num_classes``-wide
        bottleneck, which is what ``fcomb_input="logits"`` does.
        """
        if self.fcomb_input == "logits":
            return self.model.forward(img)
        # `nn.Module.__getattr__` is typed as returning `Tensor | Module`, so
        # bind these as modules before calling them.
        encoder: nn.Module = self.model.get_submodule("encoder")
        decoder: nn.Module = self.model.get_submodule("decoder")
        features = encoder(img)
        try:
            return decoder(features)
        except TypeError:
            # Older segmentation_models_pytorch decoders take the stages as
            # separate positional arguments rather than one sequence.
            return decoder(*features)

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

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the evidence lower bound (ELBO) of the log-likelihood of P(Y|X).

        Args:
            batch: batch of data with input and target key

        Returns:
            A dictionary containing the total loss,
            reconstruction loss, KL loss, and the reconstruction
        """
        img, seg_mask = batch[self.input_key], batch[self.target_key]
        # The posterior encoder concatenates the mask onto the image, so it
        # needs a channel dimension. The criterion below receives the raw
        # target from the batch, not this tensor.
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.unsqueeze(1)

        self.posterior_latent_space = self.posterior.forward(img, seg_mask)
        self.prior_latent_space = self.prior.forward(img)
        self.unet_features = self.extract_features(img)

        z_posterior = self.posterior_latent_space.rsample()

        kl_loss = torch.mean(self.kl_divergence(analytic=True, z_posterior=z_posterior))

        reconstruction = self.reconstruct(
            use_posterior_mean=False, z_posterior=z_posterior
        )

        rec_loss = self.criterion(reconstruction, batch[self.target_key])
        # ``rec_loss_mean`` is the per-pixel CE and ``rec_loss_sum`` the
        # per-image pixel sum, whatever reduction the criterion uses. Summing
        # an already mean-reduced scalar, as this used to do, returned the mean
        # again under a name that says "sum" -- a trap for anyone using it to
        # convert a paper beta (which weights a pixel sum) into a library beta.
        batch_size = reconstruction.shape[0]
        # Predicted positions per image, i.e. H*W for a [B, C, H, W] map.
        pixels_per_image = reconstruction.numel() // (
            reconstruction.shape[0] * reconstruction.shape[1]
        )
        num_positions = batch_size * pixels_per_image
        if rec_loss.ndim == 0:
            # Already reduced: "sum" over the batch scales by batch_size, while
            # "mean" does not, so recover the per-pixel mean from each.
            reduction = getattr(self.criterion, "reduction", "mean")
            total = rec_loss if reduction == "sum" else rec_loss * num_positions
        else:
            total = torch.sum(rec_loss)
        rec_loss_mean = total / num_positions
        rec_loss_sum = total / batch_size

        # ``beta`` weights the KL against the per-image pixel SUM, as in the
        # reference, so it carries the paper's meaning -- against
        # ``rec_loss_mean`` the same value is H*W times stronger and collapses
        # the posterior. The whole objective is then divided by the pixel count
        # to keep it at per-pixel magnitude. That rescaling leaves the RATIO of
        # the two terms, and hence the meaning of beta, untouched, but without
        # it the loss is ~1e5 on a 256^2 image and the resulting gradients
        # (norm ~1e5) overflow the latent heads to NaN within a few steps,
        # whatever ``gradient_clip_val`` is set to.
        loss = (rec_loss_sum + self.beta * kl_loss) / pixels_per_image

        return {
            "loss": loss,
            "rec_loss_sum": rec_loss_sum,
            "rec_loss_mean": rec_loss_mean,
            "kl_loss": kl_loss,
            "reconstruction": reconstruction,
        }

    def kl_divergence(
        self, analytic: bool = True, z_posterior: Tensor | None = None
    ) -> Tensor:
        """Compute the KL divergence between the posterior and prior KL(Q||P).

        Args:
            analytic: calculate KL analytically or via sampling from the posterior
            z_posterior: if we use sampling to approximate KL we can sample here or
                supply a sample

        Returns:
            The KL divergence
        """
        if analytic:
            # TODO this should not be necessary anymore to add this to torch source
            # see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(
                self.posterior_latent_space, self.prior_latent_space
            )
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def reconstruct(
        self, use_posterior_mean: bool = False, z_posterior: Tensor | None = None
    ) -> Tensor:
        """Reconstruct a segmentation from a posterior sample.

        Decoding a posterior sample and UNet feature map

        Args:
            use_posterior_mean: use posterior_mean instead of sampling z_q
            z_posterior: use a provided sample or sample from posterior latent space

        Returns:
            The reconstructed segmentation
        """
        if use_posterior_mean:
            # ``.mean``, not ``.loc``: the latent space is an ``Independent``
            # wrapper around ``Normal``, and only the inner ``Normal`` has
            # ``.loc`` -- reading it here raised AttributeError.
            z_posterior = self.posterior_latent_space.mean
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def sample(self, testing: bool = False) -> Tensor:
        """Sample a segmentation via reconstructing from a prior sample.

        Args:
            testing: whether to sample from the prior or use the mean
        """
        if testing is False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here.
            # For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

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

        bs = batch[self.input_key].shape[0]

        self.log("train_loss", loss_dict["loss"], batch_size=bs)
        self.log("train_rec_loss_sum", loss_dict["rec_loss_sum"], batch_size=bs)
        self.log("train_rec_loss_mean", loss_dict["rec_loss_mean"], batch_size=bs)
        self.log("train_kl_loss", loss_dict["kl_loss"], batch_size=bs)

        # compute metrics with reconstruction
        self.train_metrics(
            loss_dict["reconstruction"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        # return loss to optimize
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

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

        bs = batch[self.input_key].shape[0]

        self.log("val_loss", loss_dict["loss"], batch_size=bs)
        self.log("val_rec_loss_sum", loss_dict["rec_loss_sum"], batch_size=bs)
        self.log("val_rec_loss_mean", loss_dict["rec_loss_mean"], batch_size=bs)
        self.log("val_kl_loss", loss_dict["kl_loss"], batch_size=bs)
        # compute metrics with reconstruction
        self.val_metrics(
            loss_dict["reconstruction"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
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
            preds["pred"],
            batch[self.target_key],
            batch_size=batch[self.input_key].shape[0],
        )

        preds = self.add_aux_data_to_dict(preds, batch)

        preds[self.target_key] = batch[self.target_key]

        return preds

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if self.save_preds and not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Compute and return the prediction.

        Args:
            X: the input image
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dict
        """
        # this internally computes the latent space and unet features
        self.prior_latent_space = self.prior.forward(X)
        self.unet_features = self.extract_features(X)

        # which can then be used to sample a segmentation
        samples = torch.stack(
            [self.sample(testing=True) for _ in range(self.num_samples)], dim=-1
        )  # shape: (batch_size, num_classes, height, width, num_samples)

        return process_segmentation_prediction(samples, task=self.task)

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
