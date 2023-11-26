# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Conv Variational Layer."""

import pytest
import torch
from _pytest.fixtures import SubRequest

from lightning_uq_box.models.bnn_layers import (
    Conv1dVariational,
    Conv2dVariational,
    Conv3dVariational,
)


class TestConvVariational:
    @pytest.fixture(
        params=[
            (Conv1dVariational, "reparameterization"),
            (Conv1dVariational, "flipout"),
        ]
    )
    def conv1d_layer(self, request: SubRequest) -> Conv1dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3,),
            bias=True,
            layer_type=layer_type,
        )

    def test_forward_conv1d_layer(self, conv1d_layer: Conv1dVariational) -> None:
        """Test forward pass of Conv Variational."""
        x = torch.randn(5, 3, 3)
        out = conv1d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == conv1d_layer.out_channels

    @pytest.fixture(
        params=[
            (Conv2dVariational, "reparameterization"),
            (Conv2dVariational, "flipout"),
        ]
    )
    def conv2d_layer(self, request: SubRequest) -> Conv2dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3, 3),
            bias=True,
            layer_type=layer_type,
        )

    def test_forward_conv2d_layer(self, conv2d_layer: Conv2dVariational) -> None:
        """Test forward pass of Conv Variational."""
        x = torch.randn(5, 3, 32, 32)
        out = conv2d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == conv2d_layer.out_channels

    @pytest.fixture(
        params=[
            (Conv3dVariational, "reparameterization"),
            (Conv3dVariational, "flipout"),
        ]
    )
    def conv3d_layer(self, request: SubRequest) -> Conv3dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3, 3, 3),
            bias=True,
            layer_type=layer_type,
        )

    def test_forward_conv3d_layer(self, conv3d_layer: Conv3dVariational) -> None:
        """Test forward pass of Conv Variational ."""
        x = torch.randn(5, 3, 32, 32, 32)
        out = conv3d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == conv3d_layer.out_channels
