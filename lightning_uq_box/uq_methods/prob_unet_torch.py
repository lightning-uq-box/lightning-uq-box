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
# Licensed under the MIT License.
# Changes
# - Removed all references to tensorflow
# - adapt to lightning training framework
# - https://arxiv.org/pdf/1905.13077.pdf paper

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _HierarchicalCore(nn.Module):
    """A U-Net encoder-decoder with a full encoder and a truncated decoder.

    The truncated decoder is interleaved with the hierarchical latent space and
    has as many levels as there are levels in the hierarchy plus one additional
    level.
    """

    def __init__(
        self,
        latent_dims: List[int],
        channels_per_block: List[int],
        down_channels_per_block: Optional[List[int]] = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
        name: str = "HierarchicalDecoderDist",
    ) -> None:
        """Initializes a HierarchicalCore.

        Args:
          latent_dims: List of integers specifying the dimensions of the latents at
            each scale. The length of the list indicates the number of U-Net decoder
            scales that have latents.
          channels_per_block: A list of integers specifying the number of output
            channels for each encoder block.
          down_channels_per_block: A list of integers specifying the number of
            intermediate channels for each encoder block or None. If None, the
            intermediate channels are chosen equal to channels_per_block.
          activation_fn: A callable activation function.
          convs_per_block: An integer specifying the number of convolutional layers.
          blocks_per_level: An integer specifying the number of residual blocks per
            level.
          name: A string specifying the name of the module.
        """
        super().__init__()
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            self._down_channels_per_block = channels_per_block
        else:
            self._down_channels_per_block = down_channels_per_block
        self._name = name

    def forward(
        self,
        inputs: Tensor,
        mean: Union[bool, List[bool]] = False,
        z_q: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Forward pass through the HierarchicalCore.

        Args:
            inputs: The input tensor.
            mean: A boolean or list of booleans indicating whether to use the mean
                of the latent distributions. If a single boolean is provided, it is
                used for all levels. If a list is provided, it must have the same
                length as the number of latent levels.
            z_q: An optional tensor representing the latents.

        Returns:
            A dictionary containing the final decoder features, a list of all encoder
            outputs, a list of distributions, and a list of used latents.
        """
        encoder_features = inputs
        encoder_outputs = []
        num_levels = len(self._channels_per_block)
        num_latent_levels = len(self._latent_dims)
        if isinstance(mean, bool):
            mean = [mean] * num_latent_levels
        distributions = []
        used_latents = []

        for level in range(num_levels):
            for _ in range(self._blocks_per_level):
                encoder_features = res_block(
                    input_features=encoder_features,
                    n_channels=self._channels_per_block[level],
                    n_down_channels=self._down_channels_per_block[level],
                    activation_fn=self._activation_fn,
                    initializers=self._initializers,
                    regularizers=self._regularizers,
                    convs_per_block=self._convs_per_block,
                )

            encoder_outputs.append(encoder_features)
            if level != num_levels - 1:
                encoder_features = resize_down(encoder_features, scale=2)

        decoder_features = encoder_outputs[-1]
        for level in range(num_latent_levels):
            latent_dim = self._latent_dims[level]
            mu_logsigma = nn.Conv2d(
                decoder_features.size(1), 2 * latent_dim, (1, 1), padding="SAME"
            )(decoder_features)

            mu = mu_logsigma[..., :latent_dim]
            logsigma = mu_logsigma[..., latent_dim:]
            dist = torch.distributions.MultivariateNormal(
                mu, torch.diag_embed(torch.exp(logsigma))
            )
            distributions.append(dist)

            if z_q is not None:
                z = z_q[level]
            elif mean[level]:
                z = dist.mean
            else:
                z = dist.sample()
            used_latents.append(z)

            decoder_output_lo = torch.cat([z, decoder_features], dim=-1)
            decoder_output_hi = resize_up(decoder_output_lo, scale=2)
            decoder_features = torch.cat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], dim=-1
            )

            for _ in range(self._blocks_per_level):
                decoder_features = res_block(
                    input_features=decoder_features,
                    n_channels=self._channels_per_block[::-1][level + 1],
                    n_down_channels=self._down_channels_per_block[::-1][level + 1],
                    activation_fn=self._activation_fn,
                    initializers=self._initializers,
                    regularizers=self._regularizers,
                    convs_per_block=self._convs_per_block,
                )

        return {
            "decoder_features": decoder_features,
            "encoder_features": encoder_outputs,
            "distributions": distributions,
            "used_latents": used_latents,
        }


class _StitchingDecoder(nn.Module):
    """
    A decoder that stitches together the features from the encoder and the
    hierarchical core to produce the final output.
    """

    def __init__(
        self,
        latent_dims: List[int],
        channels_per_block: List[int],
        num_classes: int,
        down_channels_per_block: Optional[List[int]] = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
    ) -> None:
        """
        Initializes a StitchingDecoder.

        Args:
          latent_dims: List of integers specifying the dimensions of the latents at
            each scale. The length of the list indicates the number of U-Net decoder
            scales that have latents.
          channels_per_block: A list of integers specifying the number of output
            channels for each decoder block.
          num_classes: An integer specifying the number of output classes.
          down_channels_per_block: A list of integers specifying the number of
            intermediate channels for each decoder block or None. If None, the
            intermediate channels are chosen equal to channels_per_block.
          activation_fn: A callable activation function.
          convs_per_block: An integer specifying the number of convolutional layers.
          blocks_per_level: An integer specifying the number of residual blocks per
            level.
        """
        super().__init__()
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block

        self.logits = nn.Conv2d(
            self._channels_per_block[-1], self._num_classes, kernel_size=1
        )

    def forward(
        self, encoder_features: List[Tensor], decoder_features: Tensor
    ) -> Tensor:
        """
        Forward pass through the StitchingDecoder.

        Args:
            encoder_features: A list of tensors representing the features from the
                encoder.
            decoder_features: A tensor representing the features from the
                hierarchical core.

        Returns:
            A tensor representing the final output of the decoder.
        """
        num_latents = len(self._latent_dims)
        start_level = num_latents + 1
        num_levels = len(self._channels_per_block)

        for level in range(start_level, num_levels, 1):
            decoder_features = resize_up(decoder_features, scale=2)
            decoder_features = torch.cat(
                [decoder_features, encoder_features[::-1][level]], dim=1
            )
            for _ in range(self._blocks_per_level):
                decoder_features = res_block(
                    input_features=decoder_features,
                    n_channels=self._channels_per_block[::-1][level],
                    n_down_channels=self._down_channels_per_block[::-1][level],
                    activation_fn=self._activation_fn,
                    convs_per_block=self._convs_per_block,
                )

        return self.logits(decoder_features)


class HierarchicalProbUNet(nn.Module):
    """
    A Probabilistic U-Net with a hierarchical latent space. The U-Net consists of
    an encoder, a hierarchical core, and a stitching decoder.
    """

    def __init__(
        self,
        latent_dims: Tuple[int, ...] = (1, 1, 1, 1),
        channels_per_block: Optional[Tuple[int, ...]] = None,
        num_classes: int = 2,
        down_channels_per_block: Optional[Tuple[int, ...]] = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
        loss_kwargs: Optional[Dict[str, Union[float, str, bool, None]]] = None,
        name: str = "HPUNet",
    ) -> None:
        """
        Initializes a HierarchicalProbUNet.

        Args:
          latent_dims: A tuple of integers specifying the dimensions of the latents at
            each scale. The length of the tuple indicates the number of U-Net decoder
            scales that have latents.
          channels_per_block: A tuple of integers specifying the number of output
            channels for each block or None. If None, the default values are used.
          num_classes: An integer specifying the number of output classes.
          down_channels_per_block: A tuple of integers specifying the number of
            intermediate channels for each block or None. If None, the intermediate
            channels are chosen equal to channels_per_block.
          activation_fn: A callable activation function.
          convs_per_block: An integer specifying the number of convolutional layers.
          blocks_per_level: An integer specifying the number of residual blocks per
            level.
          loss_kwargs: A dictionary specifying the parameters for the loss function or
            None. If None, the default values are used.
          name: A string specifying the name of the module.
        """
        super().__init__()
        base_channels = 24
        default_channels_per_block = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple([i / 2 for i in default_channels_per_block])

        if loss_kwargs is None:
            self._loss_kwargs = {
                "type": "geco",
                "top_k_percentage": 0.02,
                "deterministic_top_k": False,
                "kappa": 0.05,
                "decay": 0.99,
                "rate": 1e-2,
                "beta": None,
            }
        else:
            self._loss_kwargs = loss_kwargs
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block

        self._prior = _HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="prior",
        )

        self._posterior = _HierarchicalCore(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="posterior",
        )

        self._f_comb = _StitchingDecoder(
            latent_dims=latent_dims,
            channels_per_block=channels_per_block,
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="f_comb",
        )

        if self._loss_kwargs["type"] == "geco":
            self._moving_average = MovingAverage(
                decay=self._loss_kwargs["decay"], differentiable=True, name="ma_test"
            )
            self._lagmul = LagrangeMultiplier(rate=self._loss_kwargs["rate"])
        self._cache = ()

    def forward(self, seg: torch.Tensor, img: torch.Tensor) -> None:
        """
        Forward pass through the model.

        Args:
            seg (torch.Tensor): The segmentation tensor.
            img (torch.Tensor): The image tensor.
        """
        inputs = (seg, img)
        if self._cache == inputs:
            return
        else:
            self._q_sample = self._posterior(torch.cat([seg, img], dim=-1), mean=False)
            self._q_sample_mean = self._posterior(
                torch.cat([seg, img], dim=-1), mean=True
            )
            self._p_sample = self._prior(img, mean=False, z_q=None)
            self._p_sample_z_q = self._prior(img, z_q=self._q_sample["used_latents"])
            self._p_sample_z_q_mean = self._prior(
                img, z_q=self._q_sample_mean["used_latents"]
            )
            self._cache = inputs
            return

    def sample(
        self, img: torch.Tensor, mean: bool = False, z_q: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Sample from the model.

        Args:
            img (torch.Tensor): The image tensor.
            mean (bool, optional): Whether to use the mean. Defaults to False.
            z_q (torch.Tensor, optional): Latent tensor. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing encoder and decoder features.
        """
        prior_out = self._prior(img, mean, z_q)
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self._f_comb(
            encoder_features=encoder_features, decoder_features=decoder_features
        )

    def reconstruct(
        self, seg: torch.Tensor, img: torch.Tensor, mean: bool = False
    ) -> Dict[str, Any]:
        """
        Reconstruct the input.

        Args:
            seg (torch.Tensor): The segmentation tensor.
            img (torch.Tensor): The image tensor.
            mean (bool, optional): Whether to use the mean. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing encoder and decoder features.
        """
        self.forward(seg, img)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self._f_comb(
            encoder_features=encoder_features, decoder_features=decoder_features
        )

    def rec_loss(
        self,
        seg: torch.Tensor,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        top_k_percentage: Optional[float] = None,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate the reconstruction loss.

        Args:
            seg (torch.Tensor): The segmentation tensor.
            img (torch.Tensor): The image tensor.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
            top_k_percentage (float, optional): The top k percentage. Defaults to None.
            deterministic (bool, optional): Whether the operation is deterministic. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the reconstruction loss.
        """
        reconstruction = self.reconstruct(seg, img, mean=False)
        return ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl(self, seg: torch.Tensor, img: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Calculate the KL divergence.

        Args:
            seg (torch.Tensor): The segmentation tensor.
            img (torch.Tensor): The image tensor.

        Returns:
            Dict[int, torch.Tensor]: A dictionary containing the KL divergence for each level.
        """
        self.forward(seg, img)
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

    def loss(
        self, seg: torch.Tensor, img: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate the loss.

        Args:
            seg (torch.Tensor): The segmentation tensor.
            img (torch.Tensor): The image tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            Dict[str, Any]: A dictionary containing the loss and summaries.
        """
        summaries = {}
        top_k_percentage = self._loss_kwargs["top_k_percentage"]
        deterministic = self._loss_kwargs["deterministic_top_k"]
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)

        kl_dict = self.kl(seg, img)
        kl_sum = torch.sum(torch.stack([kl for _, kl in kl_dict.items()], dim=-1))

        summaries["rec_loss_mean"] = rec_loss["mean"]
        summaries["rec_loss_sum"] = rec_loss["sum"]
        summaries["kl_sum"] = kl_sum
        for level, kl in kl_dict.items():
            summaries[f"kl_{level}"] = kl

        if self._loss_kwargs["type"] == "elbo":
            loss = rec_loss["sum"] + self._loss_kwargs["beta"] * kl_sum
            summaries["elbo_loss"] = loss

        elif self._loss_kwargs["type"] == "geco":
            ma_rec_loss = self._moving_average(rec_loss["sum"])
            mask_sum_per_instance = torch.sum(rec_loss["mask"], dim=-1)
            num_valid_pixels = torch.mean(mask_sum_per_instance)
            reconstruction_threshold = self._loss_kwargs["kappa"] * num_valid_pixels

            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul(rec_constraint)
            loss = lagmul * rec_constraint + kl_sum

            summaries["geco_loss"] = loss
            summaries["ma_rec_loss_mean"] = ma_rec_loss / num_valid_pixels
            summaries["num_valid_pixels"] = num_valid_pixels
            summaries["lagmul"] = lagmul
        else:
            raise NotImplementedError(
                "Loss type {} not implemented!".format(self._loss_kwargs["type"])
            )

        return dict(supervised_loss=loss, summaries=summaries)


def res_block(
    input_features,
    n_channels,
    n_down_channels=None,
    activation_fn=F.relu,
    convs_per_block=3,
):
    """
    A pre-activated residual block.

    Args:
        input_features: A tensor of shape (b, c, h, w).
        n_channels: An integer specifying the number of output channels.
        n_down_channels: An integer specifying the number of intermediate channels.
        activation_fn: A callable activation function.
        convs_per_block: An Integer specifying the number of convolutional layers.
    Returns:
        A tensor of shape (b, c, h, w).
    """
    # Pre-activate the inputs.
    skip = input_features
    residual = activation_fn(input_features)

    # Set the number of intermediate channels that we compress to.
    if n_down_channels is None:
        n_down_channels = n_channels

    for c in range(convs_per_block):
        residual = nn.Conv2d(n_down_channels, (3, 3), padding=1)(residual)
        if c < convs_per_block - 1:
            residual = activation_fn(residual)

    incoming_channels = input_features.shape[1]
    if incoming_channels != n_channels:
        skip = nn.Conv2d(n_channels, (1, 1), padding=0)(skip)
    if n_down_channels != n_channels:
        residual = nn.Conv2d(n_channels, (1, 1), padding=0)(residual)
    return skip + residual


def resize_up(input_features: Tensor, scale: int = 2) -> Tensor:
    """
    Nearest neighbor rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w).
        scale: An integer specifying the scaling factor.
    Returns: A tensor of shape (b, c, scale * h, scale * w).
    """
    assert scale >= 1
    return F.interpolate(input_features, scale_factor=scale, mode="nearest")


def resize_down(input_features: Tensor, scale: int = 2) -> Tensor:
    """
    Average pooling rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w).
        scale: An integer specifying the scaling factor.
    Returns: A tensor of shape (b, c, h / scale, w / scale).
    """
    assert scale >= 1
    return F.avg_pool2d(input_features, kernel_size=scale)


class MovingAverage(nn.Module):
    def __init__(self, decay, differentiable=False):
        super().__init__()
        self.decay = decay
        self.differentiable = differentiable
        self.register_buffer("average", torch.zeros(1))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Update and return the moving average.

        Args:
            inputs: The new inputs.

        Returns:
            The updated moving average.
        """
        if not self.differentiable:
            inputs = inputs.detach()
        self.average = self.decay * self.average + (1 - self.decay) * inputs
        return self.average


class LagrangeMultiplier(nn.Module):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate
        self.multiplier = nn.Parameter(torch.ones(1))

    def forward(self, ma_constraint: Tensor) -> Tensor:
        """
        Return the product of the multiplier and the constraint.

        Args:
            ma_constraint: The moving average constraint.

        Returns:
            The product of the multiplier and the constraint.
        """
        return self.multiplier * ma_constraint


def _sample_gumbel(shape: torch.Size, eps: float = 1e-20) -> Tensor:
    """
    Sample a Gumbel distribution.

    Args:
        shape: The shape of the output tensor.
        eps: A small constant for numerical stability.

    Returns:
        A tensor of the specified shape sampled from the Gumbel distribution.
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def _topk_mask(score: Tensor, k: int) -> Tensor:
    """
    Create a mask for the top-k elements in a score tensor.

    Args:
        score: The score tensor.
        k: The number of top elements to mask.

    Returns:
        A binary mask tensor of the same shape as the score tensor.
    """
    _, indices = torch.topk(score, k=k)
    mask = torch.zeros_like(score)
    mask.scatter_(1, indices, 1)
    return mask


def ce_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Optional[Tensor] = None,
    top_k_percentage: Optional[float] = None,
    deterministic: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute the cross-entropy loss between logits and labels.

    Args:
        logits: The logits tensor.
        labels: The labels tensor.
        mask: An optional mask tensor.
        top_k_percentage: An optional percentage for the top-k mask.
        deterministic: A boolean indicating whether to use deterministic top-k masking.

    Returns:
        A dictionary containing the mean and sum of the cross-entropy loss, and the mask.
    """
    num_classes = logits.shape[-1]
    y_flat = logits.view(-1, num_classes)
    t_flat = labels.view(-1, num_classes)
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
