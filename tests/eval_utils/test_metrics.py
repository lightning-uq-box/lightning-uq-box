# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for UQ-Prediction metrics."""

import pytest
import torch

from lightning_uq_box.uq_methods.metrics import (
    EmpiricalCoverage,
    EmpiricalCoverageBase,
    SetSize,
)


@pytest.mark.parametrize(
    "pred_set, expected_coverage, expected_set_size",
    [
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.2, 0.3, 0.4, 0.5, 0.1],
                    [0.3, 0.4, 0.5, 0.1, 0.2],
                ]
            ),
            0.5,
            2.0,
        ),
        (
            [
                torch.tensor([2, 3]),
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 1, 2]),
                torch.tensor([2, 3, 4]),
            ],
            1.0,
            2.75,
        ),
    ],
)
def test_empirical_coverage_base(pred_set, expected_coverage, expected_set_size):
    metric = EmpiricalCoverageBase(alpha=0.1, topk=2)
    targets = torch.tensor([[2], [1], [0], [2]])

    metric.update(pred_set, targets)
    result = metric.compute()

    assert result["coverage"] == expected_coverage
    assert result["set_size"] == expected_set_size


@pytest.mark.parametrize(
    "pred_set, expected",
    [
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.2, 0.3, 0.4, 0.5, 0.1],
                    [0.3, 0.4, 0.5, 0.1, 0.2],
                ]
            ),
            0.5,
        ),
        (
            [
                torch.tensor([2, 3]),
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 1, 2]),
                torch.tensor([2, 3, 4]),
            ],
            1.0,
        ),
    ],
)
def test_empirical_coverage(pred_set, expected):
    metric = EmpiricalCoverage(alpha=0.1, topk=2)
    targets = torch.tensor([[2], [1], [0], [2]])

    metric.update(pred_set, targets)
    result = metric.compute()

    assert result.item() == expected


@pytest.mark.parametrize(
    "pred_set, expected",
    [
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.2, 0.3, 0.4, 0.5, 0.1],
                    [0.3, 0.4, 0.5, 0.1, 0.2],
                ]
            ),
            2.0,
        ),
        (
            [
                torch.tensor([2, 3]),
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 1, 2]),
                torch.tensor([2, 3, 4]),
            ],
            2.75,
        ),
    ],
)
def test_set_size(pred_set, expected):
    metric = SetSize(alpha=0.1, topk=2)
    targets = torch.tensor([[2], [1], [0], [2]])

    metric.update(pred_set, targets)
    result = metric.compute()

    assert result.item() == expected
