# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Linear Variational Layer."""

import itertools

import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from lightning_uq_box.models.bnn_layers import LinearVariational


class TestLineaerVariational:
    @pytest.fixture(
        params=itertools.product(["reparameterization", "flipout"], [True, False])
    )
    def linear_variational_layer(self, request: SubRequest) -> LinearVariational:
        """Initialize a variational layer."""
        layer_type, bias = request.param
        return LinearVariational(
            in_features=1, out_features=10, bias=bias, layer_type=layer_type
        )

    def test_forward(self, linear_variational_layer: LinearVariational) -> None:
        """Test forward pass of LinearVariational."""
        x = torch.randn(5, 1)
        out = linear_variational_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == linear_variational_layer.out_features
        print(linear_variational_layer)

    @pytest.mark.parametrize(
        "stochastic_module_names",
        [[-1], [0], [0, -1], ["fc"], ["layer4.1.conv1", "fc"]],
    )
    def test_partially_stochastic(self, stochastic_module_names) -> None:
        """Test partially stochastic forward pass."""
        cfg = OmegaConf.load("tests/configs/image_regression/bnn_vi_elbo.yaml")
        model = instantiate(cfg.model, stochastic_module_names=stochastic_module_names)

        for name, module in model.model.named_modules():
            if name in model.stochastic_module_names:
                assert "Variational" in type(module).__name__
            else:
                assert "Variational" not in type(module).__name__
