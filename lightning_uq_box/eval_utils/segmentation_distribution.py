# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Distributional segmentation metrics on hard labels, with empty IoU equal to one."""

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def _check_labels(labels: Tensor, num_classes: int) -> None:
    if labels.is_floating_point() or labels.is_complex():
        raise ValueError("Metrics require integer hard label maps, not probabilities.")
    if num_classes < 2 or labels.numel() == 0:
        raise ValueError("Require nonempty labels and num_classes >= 2.")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("Label values must be in [0, num_classes).")


def iou_with_empty_convention(
    pred: Tensor, target: Tensor, num_classes: int, ignore_background: bool = True
) -> Tensor:
    """Return IoU per leading index for hard maps [..., H, W].

    Shapes may broadcast (useful for pairwise comparisons). Average over classes,
    excluding class zero by default; each absent-in-both class scores one.
    """
    _check_labels(pred, num_classes)
    _check_labels(target, num_classes)
    if pred.ndim < 2 or target.ndim < 2 or pred.shape[-2:] != target.shape[-2:]:
        raise ValueError("Hard maps must have matching spatial dimensions.")
    scores = []
    for label in range(int(ignore_background), num_classes):
        p, t = pred == label, target == label
        intersection = (p & t).sum(dim=(-2, -1))
        union = (p | t).sum(dim=(-2, -1))
        scores.append(
            torch.where(union == 0, 1.0, intersection.float() / union.clamp_min(1))
        )
    return torch.stack(scores).mean(0)


def _pairwise_iou(
    a: Tensor, b: Tensor, num_classes: int, ignore_background: bool
) -> Tensor:
    _check_labels(a, num_classes)
    _check_labels(b, num_classes)
    if a.ndim != 4 or b.ndim != 4 or a.shape[0] != b.shape[0]:
        raise ValueError("Expected [B, num_maps, H, W] with matching batch sizes.")
    # Iterate one sample axis to avoid materializing [B, N, M, H, W].
    return torch.stack(
        [
            iou_with_empty_convention(x[:, None], b, num_classes, ignore_background)
            for x in a.unbind(1)
        ],
        dim=1,
    )


def generalized_energy_distance(
    samples: Tensor, graders: Tensor, num_classes: int, ignore_background: bool = True
) -> dict[str, Tensor]:
    """Compute squared GED and its three per-image terms using 1 - IoU.

    Args:
        samples: Hard label maps [B, N, H, W].
        graders: Hard grader maps [B, M, H, W].
        num_classes: Number of classes including background.
        ignore_background: Exclude class zero from IoU.

    Returns:
        V-statistic GED squared (key ``ged``), ``d_sy``, ``d_ss``, ``d_yy``,
        each [B]. Self-pairs are included. Always report diversity ``d_ss``
        alongside GED: increasing diversity alone can improve GED.
    """
    d_sy = (1 - _pairwise_iou(samples, graders, num_classes, ignore_background)).mean(
        (1, 2)
    )
    d_ss = (1 - _pairwise_iou(samples, samples, num_classes, ignore_background)).mean(
        (1, 2)
    )
    d_yy = (1 - _pairwise_iou(graders, graders, num_classes, ignore_background)).mean(
        (1, 2)
    )
    return {"ged": 2 * d_sy - d_ss - d_yy, "d_sy": d_sy, "d_ss": d_ss, "d_yy": d_yy}


def hungarian_matched_iou(
    samples: Tensor, graders: Tensor, num_classes: int, ignore_background: bool = True
) -> Tensor:
    """Return optimal balanced IoU [B] for hard maps [B, N/M, H, W].

    Repeat each set to their least common multiple so every grader has equal
    weight, including non-divisible sample counts. Assignment runs on CPU and
    is intended for evaluation, not differentiable training.
    """
    iou = _pairwise_iou(samples, graders, num_classes, ignore_background)
    n, m = iou.shape[1:]
    size = math.lcm(n, m)
    balanced = iou.repeat_interleave(size // n, dim=1).repeat_interleave(
        size // m, dim=2
    )
    scores = []
    for matrix in balanced:
        rows, cols = linear_sum_assignment((1 - matrix).detach().cpu().numpy())
        scores.append(
            matrix[
                torch.as_tensor(rows, device=matrix.device),
                torch.as_tensor(cols, device=matrix.device),
            ].mean()
        )
    return torch.stack(scores)


def reconstruction_iou(
    pred: Tensor, target: Tensor, num_classes: int, ignore_background: bool = True
) -> Tensor:
    """Return posterior reconstruction IoU_rec [B] for hard maps [B, H, W]."""
    return iou_with_empty_convention(pred, target, num_classes, ignore_background)
