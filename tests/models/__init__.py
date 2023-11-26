# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test MLP Model."""

import torch

from lightning_uq_box.models import MLP


def test_mlp_model() -> None:
    """Test mlp model forward pass."""
    mlp = MLP()
    x_input = torch.randn(5, 1)
    out = mlp(x_input)
    assert out.shape[-1] == 1
