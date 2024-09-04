# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted from Reference Implementation: https://github.com/yookoon/density_uncertainty_layers/blob/ac47cf178938a9eaa63db682daff330c8b34fb9e/layers.py#L270

"""Masked Convolutional Layer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def causal_mask(
    width: int, height: int, starting_point: tuple[int, int]
) -> torch.Tensor:
    """Generate a causal mask for a given width, height, and starting point.

    Args:
        width: Width of the mask.
        height: Height of the mask.
        starting_point: Starting point for the mask.

    Returns:
        A tensor representing the causal mask.
    """
    row_grid, col_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    mask = np.logical_or(
        row_grid < starting_point[0],
        np.logical_and(row_grid == starting_point[0], col_grid <= starting_point[1]),
    )
    return torch.tensor(mask, dtype=torch.float32)


def conv_mask(width: int, height: int, include_center: bool = False) -> torch.Tensor:
    """Generate a convolutional mask for a given width, height, and center inclusion flag.

    Args:
        width: Width of the mask.
        height: Height of the mask.
        include_center: Whether to include the center point in the mask.

    Returns:
        A tensor representing the convolutional mask.
    """
    return causal_mask(
        width, height, starting_point=(width // 2, height // 2 + include_center - 1)
    )


def weight_mask(in_channels: int, kernel_size: tuple[int]) -> torch.Tensor:
    """Generate a weight mask for a given number of input channels and kernel size.

    Args:
        in_channels: Number of input channels.
        kernel_size: Size of the kernel.

    Returns:
        A tensor representing the weight mask.
    """
    conv_mask_with_center = conv_mask(*kernel_size, include_center=True)
    conv_mask_no_center = conv_mask(*kernel_size, include_center=False)

    mask = torch.zeros(in_channels, in_channels, *kernel_size)
    for i in range(in_channels):
        for j in range(in_channels):
            mask[i, j] = conv_mask_no_center if j >= i else conv_mask_with_center
    return mask


class MaskedConv2d(nn.Module):
    """Masked 2D Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize a MaskedConv2d layer.

        Args:
            in_channels: Number of input channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
        """
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="replicate",
        )
        self.conv.weight.data.zero_()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        mask = weight_mask(in_channels, kernel_size)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor, detach: bool = False) -> Tensor:
        """Forward pass of the masked convolutional layer.

        Args:
            x: Input tensor.
            detach: If True, detach the weights.

        Returns:
            Output tensor after applying the masked convolution.
        """
        self.conv.weight.data *= self.mask
        if detach:
            return F.conv2d(
                x,
                self.conv.weight.detach(),
                padding=self.padding,
                stride=self.stride,
                padding_mode="replicate",
            )
        else:
            return self.conv(x)
