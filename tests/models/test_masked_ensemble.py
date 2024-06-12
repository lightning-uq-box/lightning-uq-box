# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for Masked Ensemble layers."""

import pytest
import torch

from lightning_uq_box.models.masked_ensemble import MaskedConv2d, MaskedLinear


class TestMaskedLinear:
    @pytest.fixture(params=[(2, 1), (2, 2), (3, 1), (3, 2)])
    def masked_linear(self, request):
        num_estimators, scale = request.param
        return MaskedLinear(
            in_features=20,
            out_features=5,
            num_estimators=num_estimators,
            scale=scale,
            bias=True,
        )

    def test_initialization(self, masked_linear):
        assert isinstance(masked_linear, MaskedLinear)

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_forward(self, masked_linear, batch_size):
        num_estimators = masked_linear.num_estimators
        inputs = torch.randn(batch_size * num_estimators, 20)
        output = masked_linear(inputs)
        assert output.shape == (batch_size * num_estimators, 5)

    def test_invalid_batch_size(self, masked_linear):
        num_estimators = masked_linear.num_estimators
        invalid_batch_size = num_estimators + 1
        inputs = torch.randn(invalid_batch_size, 20)
        with pytest.raises(RuntimeError):
            masked_linear(inputs)

    def test_invalid_parameters(self):
        with pytest.raises(AssertionError):
            MaskedLinear(
                in_features=9, out_features=5, num_estimators=3, scale=7, bias=True
            )

    def test_print(self, masked_linear):
        print(masked_linear)


class TestMaskedConv2d:
    @pytest.fixture(params=[(2, 1), (2, 2), (3, 1), (3, 2)])
    def masked_conv2d(self, request):
        num_estimators, scale = request.param
        return MaskedConv2d(
            num_estimators=num_estimators,
            scale=scale,
            in_channels=20,
            out_channels=5,
            kernel_size=(3, 3),
            bias=True,
        )

    def test_initialization(self, masked_conv2d):
        assert isinstance(masked_conv2d, MaskedConv2d)

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_forward(self, masked_conv2d, batch_size):
        num_estimators = masked_conv2d.num_estimators
        inputs = torch.randn(batch_size * num_estimators, 20, 32, 32)
        output = masked_conv2d(inputs)
        assert output.shape == (batch_size * num_estimators, 5, 30, 30)

    def test_invalid_batch_size(self, masked_conv2d):
        num_estimators = masked_conv2d.num_estimators
        invalid_batch_size = num_estimators + 1
        inputs = torch.randn(invalid_batch_size, 20, 32, 32)
        with pytest.raises(RuntimeError):
            masked_conv2d(inputs)

    def test_invalid_input(self, masked_conv2d):
        num_estimators = masked_conv2d.num_estimators
        inputs = torch.randn(num_estimators, 20, 32, 32, 32)
        with pytest.raises(ValueError):
            masked_conv2d(inputs)

    def test_invalid_parameters(self):
        with pytest.raises(AssertionError):
            MaskedConv2d(
                num_estimators=3,
                scale=7,
                in_channels=9,
                out_channels=5,
                kernel_size=(3, 3),
                bias=True,
            )

    def test_print(self, masked_conv2d):
        print(masked_conv2d)
