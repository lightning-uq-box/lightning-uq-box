# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for Mixture Density Layer."""

import pytest
import torch

from lightning_uq_box.models.mixture_density import MixtureDensityLayer
from lightning_uq_box.models.mlp import MLP
from lightning_uq_box.uq_methods.mixture_density import MDNRegression


@pytest.fixture
def input_tensor():
    return torch.randn(10, 5)  # batch_size=10, dim_in=5


@pytest.mark.parametrize(
    "dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level",
    [
        (5, 3, 4, [10], "diagonal", None),
        (5, 3, 4, [10], "isotropic", None),
        (5, 3, 4, [10], "isotropic_clusters", None),
        (5, 3, 4, [10], "fixed", 0.1),
        (5, 3, 6, [20], "diagonal", None),
        (5, 3, 8, [30], "isotropic", None),
        (10, 5, 4, [10], "isotropic_clusters", None),
        (10, 5, 4, [10], "fixed", 0.1),
    ],
)
class TestMixtureDensityLayer:
    def test_initialization(
        self, dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level
    ):
        layer = MixtureDensityLayer(
            dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level
        )
        assert layer.dim_in == dim_in
        assert layer.dim_out == dim_out
        assert layer.n_components == n_components
        assert layer.noise_type == noise_type
        assert layer.fixed_noise_level == fixed_noise_level

    def test_forward(
        self, dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level
    ):
        input_tensor = torch.randn(10, dim_in)  # batch_size=10

        layer = MixtureDensityLayer(
            dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level
        )
        log_pi, mu, sigma = layer(input_tensor)

        assert log_pi.shape == (input_tensor.shape[0], n_components)
        assert mu.shape == (input_tensor.shape[0], n_components, dim_out)
        assert sigma.shape == (input_tensor.shape[0], n_components, dim_out)

        assert torch.allclose(
            torch.sum(torch.exp(log_pi), dim=-1),
            torch.ones(input_tensor.shape[0]),
            atol=1e-6,
        )

        if noise_type == "fixed":
            assert torch.all(sigma == fixed_noise_level)

    def test_sample_with_model(
        self, dim_in, dim_out, n_components, hidden_dims, noise_type, fixed_noise_level
    ):
        network = MLP(n_inputs=dim_in, n_outputs=dim_out)
        mixture_density_network = MDNRegression(
            model=network,
            n_components=n_components,
            hidden_dims=hidden_dims,
            noise_type=noise_type,
            fixed_noise_level=fixed_noise_level,
        )

        X = torch.randn(10, dim_in)  # batch_size=10
        samples = mixture_density_network.sample(X)
        assert samples.shape[0] == 10
        assert samples.shape[1] == dim_out


def test_invalid_noise_type():
    with pytest.raises(AssertionError):
        MixtureDensityLayer(5, 3, 4, 10, "invalid_noise_type")


def test_fixed_noise_level_mismatch():
    with pytest.raises(AssertionError):
        MixtureDensityLayer(5, 3, 4, 10, "fixed", None)
    with pytest.raises(AssertionError):
        MixtureDensityLayer(5, 3, 4, 10, "diagonal", 0.1)
