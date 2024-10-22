# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for evaluation of UQ-Predictions."""

import torch

from lightning_uq_box.eval_utils import compute_empirical_coverage


def test_compute_empirical_coverage():
    quantile_preds = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [3.0, 5.0, 7.0], [4.0, 6.0, 8.0]]
    )
    targets = torch.tensor([[3.0], [4.0], [1.0], [10.0]])

    coverage = compute_empirical_coverage(quantile_preds, targets)
    assert coverage.item() == 0.5
