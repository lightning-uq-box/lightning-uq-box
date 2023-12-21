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
# - make compatible with lightning implementation

"""Model Parts for Hierarchical Probabilistic U-Net."""

from typing import Callable, Dict, List, Optional, Union

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
    ) -> None:
        """Initializes a HierarchicalCore.

        Args:
            latent_dims: List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net decoder
                scales that have latents
            channels_per_block: A list of integers specifying the number of
                channels for each encoder block, should begin with input channel size
            down_channels_per_block: A list of integers specifying the number of
                intermediate channels for each encoder block or None. If None, the
                intermediate channels are chosen equal to channels_per_block
            activation_fn: A callable activation function
            convs_per_block: An integer specifying the number of convolutional layers.
            blocks_per_level: An integer specifying the number of residual blocks per
                level
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            self.down_channels_per_block = channels_per_block
        else:
            self.down_channels_per_block = down_channels_per_block

        # Initialize ResBlock instances for encoder
        self.encoder_res_blocks = nn.ModuleList()

        for level in range(len(self.channels_per_block)):
            in_channels = self.channels_per_block[level]
            out_channels = (
                self.channels_per_block[level + 1]
                if level < len(self.channels_per_block) - 1
                else self.channels_per_block[level]
            )
            level_res_blocks = nn.Sequential(
                *[
                    ResBlock(
                        in_channels=in_channels if block == 0 else out_channels,
                        out_channels=out_channels,
                        n_down_channels=self.down_channels_per_block[level],
                        activation_fn=self.activation_fn,
                        convs_per_block=self.convs_per_block,
                    )
                    for block in range(self.blocks_per_level)
                ]
            )
            self.encoder_res_blocks.append(level_res_blocks)

        # Initialize ResBlock instances for decoder
        self.decoder_res_blocks = nn.ModuleList()
        for level in range(len(self.latent_dims) - 1, -1, -1):
            level = level - len(self.latent_dims)
            in_channels = (
                self.channels_per_block[level]
                + self.channels_per_block[level - 1]
                + self.latent_dims[::-1][level]
            )
            out_channels = self.channels_per_block[level - 1]
            level_res_blocks = nn.Sequential(
                *[
                    ResBlock(
                        in_channels=in_channels if block == 0 else out_channels,
                        out_channels=out_channels,
                        n_down_channels=self.down_channels_per_block[level],
                        activation_fn=self.activation_fn,
                        convs_per_block=self.convs_per_block,
                    )
                    for block in range(self.blocks_per_level)
                ]
            )
            self.decoder_res_blocks.append(level_res_blocks)

        self.mu_logsigma_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 2 * latent_dim, (1, 1), padding="same")
                for in_channels, latent_dim in zip(
                    self.channels_per_block[::-1], self.latent_dims
                )
            ]
        )

    def forward(
        self,
        inputs: Tensor,
        mean: Union[bool, List[bool]] = False,
        z_q: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
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
        # TODO: this -1 is a hack?
        num_levels = len(self.channels_per_block) - 1
        num_latent_levels = len(self.latent_dims)
        if isinstance(mean, bool):
            mean = [mean] * num_latent_levels
        distributions = []
        used_latents = []

        for level in range(num_levels):
            encoder_features = self.encoder_res_blocks[level](encoder_features)

            encoder_outputs.append(encoder_features)
            if level != num_levels - 1:
                encoder_features = resize_down(encoder_features, scale=2)

        decoder_features = encoder_outputs[-1]
        for level in range(num_latent_levels):
            latent_dim = self.latent_dims[level]
            # TODO need to also configure the mu_logsigma layer as nn.ModuleList
            mu_logsigma = self.mu_logsigma_layers[level](decoder_features)

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

            # Concat and upsample the latents with the previous features.
            decoder_output_lo = torch.cat([z, decoder_features], dim=1)
            decoder_output_hi = resize_up(decoder_output_lo, scale=2)

            decoder_features = torch.cat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], dim=1
            )
            decoder_features = self.decoder_res_blocks[level](decoder_features)

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
        latent_dims: List[int],
        channels_per_block: List[int],
        num_classes: int,
        down_channels_per_block: Optional[List[int]] = None,
        activation_fn: Callable[[Tensor], Tensor] = F.relu,
        convs_per_block: int = 3,
        blocks_per_level: int = 3,
    ) -> None:
        """Initializes a StitchingDecoder.

        Args:
            latent_dims: List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net decoder
                scales that have latents
            channels_per_block: A list of integers specifying the number of
                channels for each decoder block, should begin with input channel size
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
        self.latent_dims = latent_dims
        self.channels_per_block = channels_per_block
        self.num_classes = num_classes
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block
        self.blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self.down_channels_per_block = down_channels_per_block

        self.logits = nn.Conv2d(
            self.channels_per_block[0], self.num_classes, kernel_size=1
        )

        # Initialize ResBlock instances
        self.res_blocks = nn.ModuleList()
        for level in range(len(self.channels_per_block)):
            in_channels = self.channels_per_block[::-1][level]
            out_channels = (
                self.channels_per_block[::-1][level - 1]
                if level > 0
                else self.num_classes
            )
            for _ in range(self.blocks_per_level):
                self.res_blocks.append(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        n_down_channels=self.down_channels_per_block[::-1][level],
                        activation_fn=self.activation_fn,
                        convs_per_block=self.convs_per_block,
                    )
                )

    def forward(
        self, encoder_features: List[Tensor], decoder_features: Tensor
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
        num_latents = len(self.latent_dims)
        start_level = num_latents + 1
        num_levels = len(self.channels_per_block)

        res_block_idx = 0
        for level in range(start_level, num_levels, 1):
            decoder_features = resize_up(decoder_features, scale=2)
            decoder_features = torch.cat(
                [decoder_features, encoder_features[::-1][level]], dim=1
            )
            for _ in range(self.blocks_per_level):
                decoder_features = self.res_blocks[res_block_idx](decoder_features)
                res_block_idx += 1

        return self.logits(decoder_features)


class ResBlock(nn.Module):
    """A pre-activated residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_down_channels: Optional[int] = None,
        activation_fn: Callable = F.relu,
        convs_per_block: int = 3,
    ):
        """Initializes a ResBlock.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            n_down_channels: The number of intermediate channels
            activation_fn: A callable activation function
            convs_per_block: The number of convolutional layers in the block
        """
        super().__init__()
        self.activation_fn = activation_fn
        self.convs_per_block = convs_per_block

        if n_down_channels is None:
            n_down_channels = out_channels

        self.convs = nn.ModuleList()
        for c in range(convs_per_block):
            if c == 0:
                self.convs.append(
                    nn.Conv2d(
                        in_channels, n_down_channels, kernel_size=(3, 3), padding=1
                    )
                )
            else:
                self.convs.append(
                    nn.Conv2d(
                        n_down_channels, n_down_channels, kernel_size=(3, 3), padding=1
                    )
                )

        self.skip_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), padding=0
        )
        self.residual_conv = nn.Conv2d(
            n_down_channels, out_channels, kernel_size=(1, 1), padding=0
        )

    def forward(self, input_features: Tensor) -> Tensor:
        """Forward pass of the ResBlock.

        Args:
            input_features (Tensor): A tensor of shape (b, c, h, w)

        Returns:
            Tensor: A tensor of shape (b, c, h, w)
        """
        skip = input_features
        residual = self.activation_fn(input_features)

        for c in range(self.convs_per_block):
            residual = self.convs[c](residual)
            if c < self.convs_per_block - 1:
                residual = self.activation_fn(residual)

        skip = self.skip_conv(skip)
        residual = self.residual_conv(residual)
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
