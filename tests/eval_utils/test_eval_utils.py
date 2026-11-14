# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for evaluation of UQ-Predictions."""

import numpy as np
import pytest
import torch
from scipy import stats

from lightning_uq_box.eval_utils import (
    compute_empirical_coverage,
    compute_quantiles_from_std,
)


def test_compute_empirical_coverage():
    quantile_preds = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [3.0, 5.0, 7.0], [4.0, 6.0, 8.0]]
    )
    targets = torch.tensor([[3.0], [4.0], [1.0], [10.0]])

    coverage = compute_empirical_coverage(quantile_preds, targets)
    assert coverage.item() == 0.5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("column_vectors", [False, True])
@pytest.mark.parametrize("stds", [[0.0, 0.0, 0.0], [0.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
def test_compute_quantiles_from_std(
    dtype: type, column_vectors: bool, stds: list[float]
) -> None:
    """Zero spread collapses the interval; positive spread retains Gaussian quantiles."""
    means = np.array([-3.0, 0.0, 2.0], dtype=dtype)
    scales = np.array(stds, dtype=dtype)
    quantiles = [0.05, 0.5, 0.8, 0.95]
    expected = means[:, None] + scales[:, None] * stats.norm.ppf(quantiles)
    if column_vectors:
        means = means[:, None]
        scales = scales[:, None]

    result = compute_quantiles_from_std(means, scales, quantiles)

    assert result.shape == (3, 4)
    np.testing.assert_allclose(result, expected)


def test_compute_quantiles_from_identical_samples() -> None:
    """Agreement between sampled predictions gives finite, zero-width intervals."""
    samples = torch.tensor([[2.0, 2.0, 2.0]])

    result = compute_quantiles_from_std(
        samples.mean(-1).numpy(), samples.std(-1).numpy(), [0.05, 0.5, 0.95]
    )

    np.testing.assert_array_equal(result, [[2.0, 2.0, 2.0]])
