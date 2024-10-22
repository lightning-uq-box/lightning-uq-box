# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for Spectral Normalization Layers."""

import pytest
import torch
import torch.nn as nn

from lightning_uq_box.uq_methods.spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralBatchNorm3d,
    spectral_norm_batch_norm,
    spectral_norm_conv,
    spectral_norm_fc,
)


@pytest.mark.parametrize(
    "module, input_tensor, expected_type",
    [
        (nn.BatchNorm1d(10), torch.randn(20, 10), SpectralBatchNorm1d),
        (nn.BatchNorm2d(10), torch.randn(20, 10, 35, 45), SpectralBatchNorm2d),
        (nn.BatchNorm3d(10), torch.randn(20, 10, 35, 45, 55), SpectralBatchNorm3d),
    ],
)
def test_spectral_norm_batch_norm(module, input_tensor, expected_type):
    sn_bn = spectral_norm_batch_norm(module, coeff=0.5)

    assert isinstance(sn_bn, expected_type)

    output = sn_bn(input_tensor)
    assert output.shape == input_tensor.shape


def test_spectral_norm_batch_norm_unsupported_type():
    linear = nn.Linear(10, 10)
    with pytest.raises(ValueError):
        spectral_norm_batch_norm(linear, coeff=0.5)


@pytest.mark.parametrize(
    "module, input_tensor, output_shape",
    [
        (
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(3, 3), padding=1),
            torch.randn(1, 3, 32, 32),
            torch.Size([1, 20, 32, 32]),
        )
    ],
)
def test_spectral_norm_conv(module, input_tensor, output_shape):
    sn_conv = spectral_norm_conv(
        module, coeff=0.5, input_dim=input_tensor.squeeze(0).shape
    )

    assert hasattr(sn_conv, "weight_u")

    output = sn_conv(input_tensor)
    assert output.shape == output_shape


def test_spectral_norm_conv_unsupported_type():
    linear = nn.Linear(10, 10)
    with pytest.raises(TypeError):
        spectral_norm_conv(linear, coeff=0.5, input_dim=1)


def test_spectral_norm_conv_stride_larger():
    conv = nn.Conv2d(
        in_channels=3, out_channels=20, kernel_size=(3, 3), padding=1, stride=3
    )
    input = torch.randn(3, 64, 64)
    spectral_norm_conv(conv, coeff=0.5, input_dim=input.shape)
    with pytest.raises(RuntimeError):
        conv(input.unsqueeze(0))


@pytest.mark.parametrize(
    "module, input_tensor", [(nn.Linear(10, 20), torch.randn(20, 10))]
)
def test_spectral_norm_fc(module, input_tensor):
    sn_fc = spectral_norm_fc(module, coeff=0.5)

    assert hasattr(sn_fc, "weight_u")

    output = sn_fc(input_tensor)
    assert output.shape[0] == input_tensor.shape[0]
