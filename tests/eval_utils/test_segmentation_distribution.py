# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

import pytest
import torch

from lightning_uq_box.eval_utils import (
    generalized_energy_distance,
    hungarian_matched_iou,
    iou_with_empty_convention,
    reconstruction_iou,
)


def test_empty_and_disjoint() -> None:
    empty = torch.zeros(2, 3, 3, dtype=torch.long)
    full = torch.ones_like(empty)
    torch.testing.assert_close(
        iou_with_empty_convention(empty, empty, 2), torch.ones(2)
    )
    torch.testing.assert_close(reconstruction_iou(empty, full, 2), torch.zeros(2))


def test_analytic_ged_and_balancing() -> None:
    samples = torch.tensor([[[[0]], [[1]]]])
    graders = samples.repeat(1, 2, 1, 1)
    result = generalized_energy_distance(samples, graders, 2)
    assert result["ged"].item() == 0
    assert (
        result["d_ss"].item() == result["d_sy"].item() == result["d_yy"].item() == 0.5
    )
    assert hungarian_matched_iou(samples, graders, 2).item() == 1
    assert hungarian_matched_iou(samples[:, :1], samples[:, 1:], 2).item() == 0
    assert (
        generalized_energy_distance(samples[:, :1], samples[:, 1:], 2)["ged"].item()
        == 2
    )
    # Two vs three maps: equal weighting requires repeating BOTH to six.
    three = torch.tensor([[[[0]], [[0]], [[1]]]])
    torch.testing.assert_close(
        hungarian_matched_iou(samples, three, 2), torch.tensor([5 / 6])
    )


def test_multiclass_and_invalid_probabilities() -> None:
    pred = torch.tensor([[[1, 1], [2, 0]]])
    target = torch.tensor([[[1, 2], [2, 0]]])
    torch.testing.assert_close(
        iou_with_empty_convention(pred, target, 3), torch.tensor([0.5])
    )
    with pytest.raises(ValueError, match="hard label"):
        generalized_energy_distance(pred[:, None].float(), target[:, None], 3)
