# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test LSTM Variational Layer."""

import itertools

import pytest
import torch
from _pytest.fixtures import SubRequest

from lightning_uq_box.models.bnn_layers import LSTMVariational


class TestLSTMVariational:
    @pytest.fixture(
        params=itertools.product(["reparameterization", "flipout"], [True, False])
    )
    def lstm_variational_layer(self, request: SubRequest) -> LSTMVariational:
        """Initialize a variational layer."""
        layer_type, bias = request.param
        return LSTMVariational(
            in_features=1, out_features=10, bias=bias, layer_type=layer_type
        )

    def test_forward(self, lstm_variational_layer: LSTMVariational) -> None:
        """Test forward pass of LSTMVariational."""
        # batch_size, sequence length, input size
        x = torch.randn(3, 5, 1)
        hidden, _ = lstm_variational_layer(x)
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape[0] == 3  # batch size
        assert hidden.shape[1] == 5  # sequence length
        assert hidden.shape[-1] == lstm_variational_layer.out_features
