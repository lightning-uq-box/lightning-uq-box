# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for Masked Convolutional Layer."""

import pytest
import torch

from lightning_uq_box.models.masked_conv import MaskedConv2d


@pytest.fixture
def input_tensor():
    """Fixture for creating a sample input tensor."""
    return torch.randn(1, 3, 8, 8)  # Batch size 1, 3 channels, 8x8 image


def test_masked_conv2d_initialization():
    """Test the initialization of the MaskedConv2d layer."""
    in_channels = 3
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    layer = MaskedConv2d(in_channels, kernel_size, stride, padding)

    assert layer.conv.in_channels == in_channels
    assert layer.conv.out_channels == in_channels
    assert layer.conv.kernel_size == kernel_size
    assert layer.conv.stride == (stride, stride)
    assert layer.conv.padding == (padding, padding)
    assert layer.conv.bias is None
    assert layer.mask.shape == (in_channels, in_channels, *kernel_size)


def test_masked_conv2d_initialization_with_int_kernel_size():
    """Test the initialization of the MaskedConv2d layer with an integer kernel size."""
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    layer = MaskedConv2d(in_channels, kernel_size, stride, padding)

    assert layer.conv.kernel_size == (kernel_size, kernel_size)
    assert layer.mask.shape == (in_channels, in_channels, kernel_size, kernel_size)


def test_masked_conv2d_forward(input_tensor):
    """Test the forward pass of the MaskedConv2d layer."""
    in_channels = 3
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    layer = MaskedConv2d(in_channels, kernel_size, stride, padding)

    output = layer(input_tensor)
    assert output.shape == input_tensor.shape
