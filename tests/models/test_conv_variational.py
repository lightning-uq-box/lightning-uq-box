# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Conv Variational Layer."""

import itertools

import pytest
import torch
from _pytest.fixtures import SubRequest

from lightning_uq_box.models.bnn_layers import (
    Conv1dVariational,
    Conv2dVariational,
    Conv3dVariational,
    ConvTranspose1dVariational,
    ConvTranspose2dVariational,
    ConvTranspose3dVariational,
)

layer_types = ["reparameterization", "flipout"]
biases = [True, False]


class TestConvVariational:
    @pytest.fixture(params=itertools.product([Conv1dVariational], layer_types, biases))
    def conv1d_layer(self, request: SubRequest) -> Conv1dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3,),
            bias=bias,
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
        params=itertools.product([ConvTranspose1dVariational], layer_types, biases)
    )
    def convtranspose1d_layer(self, request: SubRequest) -> ConvTranspose1dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=7,
            out_channels=3,
            kernel_size=(3,),
            bias=bias,
            layer_type=layer_type,
        )

    def test_forward_convtranspose1d_layer(
        self, convtranspose1d_layer: ConvTranspose1dVariational
    ) -> None:
        """Test forward pass of ConvTranspose Variational."""
        x = torch.randn(5, 3, 32)
        out = convtranspose1d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == convtranspose1d_layer.in_channels

    @pytest.fixture(params=itertools.product([Conv2dVariational], layer_types, biases))
    def conv2d_layer(self, request: SubRequest) -> Conv2dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3, 3),
            bias=bias,
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
        params=itertools.product([ConvTranspose2dVariational], layer_types, biases)
    )
    def convtranspose2d_layer(self, request: SubRequest) -> ConvTranspose2dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=7,
            out_channels=3,
            kernel_size=(3, 3),
            bias=bias,
            layer_type=layer_type,
        )

    def test_forward_convtranspose2d_layer(
        self, convtranspose2d_layer: ConvTranspose2dVariational
    ) -> None:
        """Test forward pass of ConvTranspose Variational."""
        x = torch.randn(5, 3, 32, 32)
        out = convtranspose2d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == convtranspose2d_layer.in_channels

    @pytest.fixture(params=itertools.product([Conv3dVariational], layer_types, biases))
    def conv3d_layer(self, request: SubRequest) -> Conv3dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=3,
            out_channels=7,
            kernel_size=(3, 3, 3),
            bias=bias,
            layer_type=layer_type,
        )

    def test_forward_conv3d_layer(self, conv3d_layer: Conv3dVariational) -> None:
        """Test forward pass of Conv Variational ."""
        x = torch.randn(5, 3, 32, 32, 32)
        out = conv3d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == conv3d_layer.out_channels

    @pytest.fixture(
        params=itertools.product([ConvTranspose3dVariational], layer_types, biases)
    )
    def convtranspose3d_layer(self, request: SubRequest) -> ConvTranspose3dVariational:
        """Initialize a variational layer."""
        conv_layer, layer_type, bias = request.param
        return conv_layer(
            in_channels=7,
            out_channels=3,
            kernel_size=(3, 3, 3),
            bias=bias,
            layer_type=layer_type,
        )

    def test_forward_convtranspose3d_layer(
        self, convtranspose3d_layer: ConvTranspose3dVariational
    ) -> None:
        """Test forward pass of ConvTranspose Variational."""
        x = torch.randn(5, 3, 32, 32, 32)
        out = convtranspose3d_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == convtranspose3d_layer.in_channels
