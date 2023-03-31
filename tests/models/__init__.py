"""Test MLP Model."""

import torch

from uq_method_box.models import MLP


def test_mlp_model() -> None:
    """Test mlp model forward pass."""
    mlp = MLP()
    x_input = torch.randn(5, 1)
    out = mlp(x_input)
    assert out.shape[-1] == 1
