# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test LSTM Variational Layer."""

import itertools

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from lightning_uq_box.models.bnn_layers import LSTMVariational
from lightning_uq_box.models.bnn_layers.bnn_utils import convert_deterministic_to_bnn


class TestLSTMVariational:
    @pytest.fixture(
        params=itertools.product(["reparameterization", "flipout"], [True, False])
    )
    def lstm_variational_layer(self, request: SubRequest) -> LSTMVariational:
        """Initialize a variational layer."""
        layer_type, bias = request.param
        return LSTMVariational(
            in_features=2, out_features=10, bias=bias, layer_type=layer_type
        )

    def test_forward(self, lstm_variational_layer: LSTMVariational) -> None:
        """Test forward pass of LSTMVariational."""
        # batch_size, sequence length, input size
        x = torch.randn(3, 5, 2)
        hidden, _ = lstm_variational_layer(x)
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape[0] == 3  # batch size
        assert hidden.shape[1] == 5  # sequence length
        assert hidden.shape[-1] == lstm_variational_layer.out_features

    def test_compute_loss(self, lstm_variational_layer: LSTMVariational) -> None:
        """Test compute loss of LSTMVariational."""
        # batch_size, sequence length, input size
        x = torch.randn(3, 5, 2)
        hidden, _ = lstm_variational_layer(x)
        log_Z_prior = lstm_variational_layer.calc_log_Z_prior()
        assert isinstance(log_Z_prior, torch.Tensor)
        log_f_hat = lstm_variational_layer.log_f_hat()
        assert isinstance(log_f_hat, torch.Tensor)
        log_normalizer = lstm_variational_layer.log_normalizer()
        assert isinstance(log_normalizer, torch.Tensor)

    @pytest.mark.parametrize("hidden_state", [True, False])
    def test_forward_pass(self, lstm_variational_layer, hidden_state) -> None:
        """Test forward of LSTMVariational"""
        x = torch.randn(3, 5, 2)  # batch_size, sequence length, input size
        prev_hidden_state = None
        if hidden_state:
            h0 = torch.randn(1, 3, 10)  # num_layers, batch_size, hidden_size
            c0 = torch.randn(1, 3, 10)  # num_layers, batch_size, hidden_size
            prev_hidden_state = (h0, c0)
        output1, (hn1, cn1) = lstm_variational_layer(x, prev_hidden_state)
        output2, (hn2, cn2) = lstm_variational_layer(x, prev_hidden_state)
        assert not torch.equal(output1, output2)


class TestBnnLstmConversion:
    @pytest.fixture
    def lstm_model(self) -> nn.Module:
        """Initialize a deterministic LSTM model."""
        return nn.Sequential(
            nn.LSTM(input_size=10, hidden_size=20, num_layers=2), nn.Linear(20, 10)
        )

    @pytest.fixture
    def bnn_params(self) -> dict:
        """Initialize parameters for Bayesian LSTM layer."""
        return {
            "prior_mu": 0,
            "prior_sigma": 0.1,
            "posterior_mu_init": 0,
            "posterior_rho_init": -3,
        }

    def test_conversion(self, lstm_model: nn.Module, bnn_params: dict) -> None:
        """Test conversion of deterministic LSTM model to Bayesian LSTM model."""
        convert_deterministic_to_bnn(lstm_model, bnn_params, ["0"])

        assert isinstance(lstm_model[0], LSTMVariational)
        assert isinstance(lstm_model[1], nn.Linear)

    def test_forward_pass(self, lstm_model: nn.Module, bnn_params: dict) -> None:
        """Test forward of LSTMVariational model."""
        convert_deterministic_to_bnn(lstm_model, bnn_params, ["0"])

        x = torch.randn(3, 5, 10)  # batch_size, sequence length, input size
        output1, _ = lstm_model[0](x)
        output1 = lstm_model[1](output1[-1])
        output2, _ = lstm_model[0](x)
        output2 = lstm_model[1](output2[-1])
        assert not torch.equal(output1, output2)
