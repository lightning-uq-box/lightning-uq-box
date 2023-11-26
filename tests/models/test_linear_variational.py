# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Linear Variational Layer."""

import pytest
import torch
from _pytest.fixtures import SubRequest

from lightning_uq_box.models.bnn_layers import LinearVariational


class TestLineaerVariational:
    @pytest.fixture(params=["reparameterization", "flipout"])
    def linear_variational_layer(self, request: SubRequest) -> LinearVariational:
        """Initialize a variational layer."""
        return LinearVariational(
            in_features=1, out_features=10, bias=True, layer_type=request.param
        )

    def test_forward(self, linear_variational_layer: LinearVariational) -> None:
        """Test forward pass of LinearVariational."""
        x = torch.randn(5, 1)
        out = linear_variational_layer(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 5
        assert out.shape[1] == linear_variational_layer.out_features
