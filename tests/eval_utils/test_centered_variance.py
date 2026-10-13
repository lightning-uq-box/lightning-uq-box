# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Moment matching should retain small variance around a large mean."""

import pytest
import torch

from lightning_uq_box.eval_utils import (
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
)
from lightning_uq_box.uq_methods.utils import process_regression_prediction


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_samples", [1, 3])
@pytest.mark.parametrize("predictive", [False, True])
def test_centered_mixture_uncertainty(
    dtype: torch.dtype, shifted: bool, batch_size: int, n_samples: int, predictive: bool
) -> None:
    offset = (1e4 if dtype == torch.float32 else 1e10) if shifted else 0.0
    deviations = torch.tensor([[0.0]] if n_samples == 1 else [[-1.0, 0.0, 1.0]])
    means = (
        (deviations.to(dtype) + offset).expand(batch_size, -1).clone().requires_grad_()
    )
    sigmas = torch.ones_like(means, requires_grad=True)
    variance = 0.0 if n_samples == 1 else 2 / 3
    expected = means.new_full((batch_size,), (variance + int(predictive)) ** 0.5)
    actual = (
        compute_predictive_uncertainty(means, sigmas)
        if predictive
        else compute_epistemic_uncertainty(means)
    )
    torch.testing.assert_close(actual, expected)
    if n_samples > 1 or predictive:
        # d(sqrt(var + mean(sigma^2)))) / d(mu_i) = centered_mu_i / (n * std).
        expected_grad = deviations.to(dtype).expand(batch_size, -1) / (
            n_samples * expected[:, None]
        )
        actual_grad = torch.autograd.grad(actual.sum(), means, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        if predictive:
            sigma_grad = torch.autograd.grad(actual.sum(), sigmas)[0]
            torch.testing.assert_close(
                sigma_grad, 1 / (n_samples * expected[:, None]).expand_as(sigmas)
            )


def test_single_component_nll_prediction_keeps_variance() -> None:
    predictions = torch.tensor([[[10000.0], [0.0]]])
    result = process_regression_prediction(predictions)
    torch.testing.assert_close(result["pred_uct"], torch.tensor([(1 + 1e-6) ** 0.5]))
    torch.testing.assert_close(result["epistemic_uct"], torch.tensor([0.0]))
