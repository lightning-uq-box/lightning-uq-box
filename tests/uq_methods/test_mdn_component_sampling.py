# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Categorical MDN sampling must respect probability bins and input dtypes."""

import math

import pytest
import torch
from torch import Tensor, nn

from lightning_uq_box.models.mixture_density import MixtureDensityLayer
from lightning_uq_box.uq_methods.mixture_density import MDNRegression


def fixed_uniform(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    def draw(
        *size: int, dtype: torch.dtype | None = None, device: torch.device | None = None
    ) -> Tensor:
        return torch.full(size, value, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "rand", draw)


@pytest.mark.parametrize("input_dtype", [torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "case", ["upper", "zero_mass", "interior", "boundary", "single"]
)
def test_component_sampling_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    input_dtype: torch.dtype,
    batch_size: int,
    case: str,
) -> None:
    if case == "upper":
        logits, uniform, expected = torch.zeros(27), 0.9999999403953552, 26
    elif case == "zero_mass":
        logits, uniform, expected = torch.tensor([-1000.0, 0.0, -1000.0]), 0.0, 1
    elif case == "interior":
        logits, uniform, expected = (
            torch.tensor([math.log(0.25), math.log(0.75)]),
            0.75,
            1,
        )
    elif case == "boundary":
        logits, uniform, expected = torch.zeros(2), 0.5, 1
    else:
        logits, uniform, expected = torch.zeros(1), 0.75, 0
    model = MDNRegression(nn.Sequential(nn.Linear(1, 1)), len(logits), hidden_dims=[])
    log_pi = torch.log_softmax(logits, dim=-1).expand(batch_size, -1).contiguous()
    inputs = torch.zeros(batch_size, 1, dtype=input_dtype)
    fixed_uniform(monkeypatch, uniform)
    result = model.select_mixture_component_preds(inputs, log_pi)
    torch.testing.assert_close(result, torch.full((batch_size, 1, 1), expected))


def test_embedding_prediction_samples_from_float_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = nn.Sequential(nn.Embedding(3, 2), nn.Flatten(), nn.Linear(2, 1))
    model = MDNRegression(backbone, n_components=2, hidden_dims=[])
    layer = next(
        module for module in model.modules() if isinstance(module, MixtureDensityLayer)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        pi_output = layer.pi_network.model[-1]
        normal_output = layer.normal_network.model[-1]
        assert isinstance(pi_output, nn.Linear) and pi_output.bias is not None
        assert isinstance(normal_output, nn.Linear) and normal_output.bias is not None
        pi_output.bias.copy_(torch.tensor([math.log(0.25), math.log(0.75)]))
        normal_output.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))
    fixed_uniform(monkeypatch, 0.75)
    prediction = model.predict_step(torch.tensor([[1], [2]]))
    torch.testing.assert_close(prediction["pred"], torch.full((2, 1), 10.0))
