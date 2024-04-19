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
# - make compatible with lightning implementation

"""Model Parts for Hierarchical Probabilistic U-Net."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# TODO this is a model
class _HierarchicalCore(nn.Module):
    """A U-Net encoder-decoder with a full encoder and a truncated decoder.

    The truncated decoder is interleaved with the hierarchical latent space and
    has as many levels as there are levels in the hierarchy plus one additional
    level.
    """

    def __init__(
        self,
        latent_dims: list[int],
        channels_per_block: list[int],
        down_channels_per_block: list[int] | None = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
    ) -> None:
        """Initializes a HierarchicalCore.

        Args:
            latent_dims: List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net decoder
                scales that have latents
            channels_per_block: A list of integers specifying the number of output
                channels for each encoder block
            down_channels_per_block: A list of integers specifying the number of
                intermediate channels for each encoder block or None. If None, the
                intermediate channels are chosen equal to channels_per_block
            activation_fn: A callable activation function
            convs_per_block: An integer specifying the number of convolutional layers.
            blocks_per_level: An integer specifying the number of residual blocks per
                level
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

    def forward(
        self, inputs: Tensor, mean: bool | list[bool] = False, z_q: Tensor | None = None
    ) -> dict[str, Tensor | list[Tensor]]:
        """Forward pass through the HierarchicalCore.

        Args:
            inputs: The input tensor
            mean: A boolean or list of booleans indicating whether to use the mean
                of the latent distributions. If a single boolean is provided, it is
                used for all levels. If a list is provided, it must have the same
                length as the number of latent levels
            z_q: An optional tensor representing the latents

        Returns:
            A dictionary containing the final decoder features, a list of all encoder
            outputs, a list of distributions, and a list of used latents
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
                    convs_per_block=self._convs_per_block,
                )

            encoder_outputs.append(encoder_features)
            if level != num_levels - 1:
                encoder_features = resize_down(encoder_features, scale=2)

        decoder_features = encoder_outputs[-1]
        for level in range(num_latent_levels):
            latent_dim = self._latent_dims[level]
            mu_logsigma = nn.Conv2d(
                decoder_features.size(1), 2 * latent_dim, (1, 1), padding="same"
            )(decoder_features)

            mu = mu_logsigma.narrow(1, 0, latent_dim)
            logsigma = mu_logsigma.narrow(
                1, latent_dim, mu_logsigma.size(1) - latent_dim
            )
            # mu = mu_logsigma[..., :latent_dim]
            # logsigma = mu_logsigma[..., latent_dim:]
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

            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = resize_up(decoder_output_lo, scale=2)
            decoder_features = torch.cat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], dim=1
            )

            for _ in range(self._blocks_per_level):
                decoder_features = res_block(
                    input_features=decoder_features,
                    n_channels=self._channels_per_block[::-1][level + 1],
                    n_down_channels=self._down_channels_per_block[::-1][level + 1],
                    activation_fn=self._activation_fn,
                    convs_per_block=self._convs_per_block,
                )

        return {
            "decoder_features": decoder_features,
            "encoder_features": encoder_outputs,
            "distributions": distributions,
            "used_latents": used_latents,
        }


class _StitchingDecoder(nn.Module):
    """Stitching Decoder.

    A decoder that stitches together the features from the encoder and the
    hierarchical core to produce the final output.
    """

    def __init__(
        self,
        latent_dims: list[int],
        channels_per_block: list[int],
        num_classes: int,
        down_channels_per_block: list[int] | None = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
    ) -> None:
        """Initializes a StitchingDecoder.

        Args:
            latent_dims: List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net decoder
                scales that have latents
            channels_per_block: A list of integers specifying the number of output
                channels for each decoder block
            num_classes: An integer specifying the number of output classes.
            down_channels_per_block: A list of integers specifying the number of
                intermediate channels for each decoder block or None. If None, the
                intermediate channels are chosen equal to channels_per_block.
            activation_fn: A callable activation function
            convs_per_block: An integer specifying the number of convolutional layers
            blocks_per_level: An integer specifying the number of residual blocks per
                level
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
            self._channels_per_block[0], self._num_classes, kernel_size=1
        )

    def forward(
        self, encoder_features: list[Tensor], decoder_features: Tensor
    ) -> Tensor:
        """Forward pass through the StitchingDecoder.

        Args:
            encoder_features: A list of tensors representing the features from the
                encoder
            decoder_features: A tensor representing the features from the
                hierarchical core

        Returns:
            A tensor representing the final output of the decoder
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


def res_block(
    input_features: Tensor,
    n_channels: int,
    n_down_channels=int,
    activation_fn=F.relu,
    convs_per_block: int = 3,
):
    """A pre-activated residual block.

    Args:
        input_features: A tensor of shape (b, c, h, w)
        n_channels: An integer specifying the number of output channels
        n_down_channels: An integer specifying the number of intermediate channels
        activation_fn: A callable activation function
        convs_per_block: An Integer specifying the number of convolutional layers

    Returns:
        A tensor of shape (b, c, h, w)
    """
    # Pre-activate the inputs.
    skip = input_features
    residual = activation_fn(input_features)

    # Set the number of intermediate channels that we compress to.
    if n_down_channels is None:
        n_down_channels = n_channels

    for c in range(convs_per_block):
        if c == 0 and input_features.shape[1] != n_down_channels:
            conv = nn.Conv2d(
                input_features.shape[1], n_down_channels, kernel_size=(3, 3), padding=1
            )
        else:
            conv = nn.Conv2d(
                n_down_channels, n_down_channels, kernel_size=(3, 3), padding=1
            )
        residual = conv(residual)
        if c < convs_per_block - 1:
            residual = activation_fn(residual)

    incoming_channels = input_features.shape[1]
    if incoming_channels != n_channels:
        conv = nn.Conv2d(incoming_channels, n_channels, kernel_size=(1, 1), padding=0)
        skip = conv(skip)
    if n_down_channels != n_channels:
        conv = nn.Conv2d(n_down_channels, n_channels, kernel_size=(1, 1), padding=0)
        residual = conv(residual)
    return skip + residual


def resize_up(input_features: Tensor, scale: int = 2) -> Tensor:
    """Nearest neighbor rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w)
        scale: An integer specifying the scaling factor

    Returns:
        A tensor of shape (b, c, scale * h, scale * w).
    """
    assert scale >= 1
    return F.interpolate(input_features, scale_factor=scale, mode="nearest")


def resize_down(input_features: Tensor, scale: int = 2) -> Tensor:
    """Average pooling rescaling-operation for the input features.

    Args:
        input_features: A tensor of shape (b, c, h, w).
        scale: An integer specifying the scaling factor.

    Returns:
        A tensor of shape (b, c, h / scale, w / scale).
    """
    assert scale >= 1
    return F.avg_pool2d(input_features, kernel_size=scale)


class MovingAverage(nn.Module):
    """Moving Average Compute Module."""

    def __init__(self, decay: float = 0.99, differentiable=False):
        """Initialize the MovingAverage.

        Args:
            decay: The decay of the moving average
            differentiable: Whether the moving average should be differentiable
        """
        super().__init__()
        self.decay = decay
        self.differentiable = differentiable
        self.register_buffer("average", torch.zeros(1))

    def forward(self, inputs: Tensor) -> Tensor:
        """Update and return the moving average.

        Args:
            inputs: The new inputs

        Returns:
            The updated moving average
        """
        if not self.differentiable:
            inputs = inputs.detach()
        self.average = self.decay * self.average + (1 - self.decay) * inputs
        return self.average


class LagrangeMultiplier(nn.Module):
    """Lagrange Multiplier Compute Module."""

    def __init__(self, rate=1e-2):
        """Initialize the LagrangeMultiplier.

        Args:
            rate: The rate of the Lagrange multiplier
        """
        super().__init__()
        self.rate = rate
        self.multiplier = nn.Parameter(torch.ones(1))

    def forward(self, ma_constraint: Tensor) -> Tensor:
        """Return the product of the multiplier and the constraint.

        Args:
            ma_constraint: The moving average constraint

        Returns:
            The product of the multiplier and the constraint
        """
        return self.multiplier * ma_constraint
