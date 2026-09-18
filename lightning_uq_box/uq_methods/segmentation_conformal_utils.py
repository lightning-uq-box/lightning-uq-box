# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Numerical helpers for binary, image-weighted conformal segmentation."""

import math

import torch
from torch import Tensor


def binary_segmentation_inputs(phat: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Validate probabilities and masks [B,1,H,W] or [B,H,W], returning [B,1,H,W]."""
    if phat.ndim == 3:
        phat = phat.unsqueeze(1)
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    if (
        phat.ndim != 4
        or phat.shape[1] != 1
        or labels.shape != phat.shape
        or phat.shape[0] == 0
    ):
        raise ValueError("Expected matching nonempty binary maps [B,1,H,W].")
    if not torch.isfinite(phat).all() or ((phat < 0) | (phat > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1].")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary.")
    return phat, labels.bool()


def per_image_coverage(pred_mask: Tensor, labels: Tensor) -> Tensor:
    """Return positive-pixel recall [B]; empty foreground masks have coverage one."""
    positives = labels.bool().flatten(1).sum(1)
    covered = (pred_mask.bool() & labels.bool()).flatten(1).sum(1)
    return torch.where(positives > 0, covered / positives.clamp_min(1), 1.0)


@torch.no_grad()
def compute_oracle_threshold(
    phat: Tensor,
    labels: Tensor,
    alpha: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tensor:
    """Find largest feasible per-image thresholds [B] using hard recall bisection.

    Discrete recall need not equal the target exactly. Return the conservative
    endpoint; tied probabilities can make the overshoot larger than one pixel.
    """
    if not 0 < alpha < 1 or not math.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid alpha or search settings.")
    phat, labels = binary_segmentation_inputs(phat, labels)
    low = phat.new_zeros(phat.shape[0])
    high = torch.ones_like(low)
    for _ in range(max_iter):
        mid = (low + high) / 2
        feasible = (
            per_image_coverage(phat >= mid[:, None, None, None], labels) >= 1 - alpha
        )
        low = torch.where(feasible, mid, low)
        high = torch.where(feasible, high, mid)
        if (high - low).max() <= tol:
            break
    return torch.where(per_image_coverage(phat >= 1, labels) >= 1 - alpha, 1.0, low)


@torch.no_grad()
def find_calibration_shift(
    predicted_tau: Tensor,
    phat: Tensor,
    labels: Tensor,
    target_coverage: float,
    search_range: tuple[float, float] = (-1.0, 1.0),
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Find the smallest conservative subtraction shift for split calibration.

    Maps have shape [B,1,H,W] or [B,H,W], thresholds [B]. The target is
    multiplied by (n+1)/n. If that exceeds one, return the full-foreground
    prediction (shift one). This fallback cannot satisfy the empirical corrected
    target, but has zero FNR. Wider bounds than the reference cover all sigmoid
    thresholds. Exchangeability and a held-out calibration split are required
    for the marginal guarantee; conditional coverage is not guaranteed.
    """
    if (
        not 0 < target_coverage < 1
        or not math.isfinite(tol)
        or tol <= 0
        or max_iter < 1
    ):
        raise ValueError("Invalid target coverage or search settings.")
    phat, labels = binary_segmentation_inputs(phat, labels)
    if (
        predicted_tau.shape != (phat.shape[0],)
        or not torch.isfinite(predicted_tau).all()
        or ((predicted_tau < 0) | (predicted_tau > 1)).any()
    ):
        raise ValueError("Expected finite thresholds [B] in [0,1].")
    low, high = search_range
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Invalid search range.")
    target = target_coverage * (phat.shape[0] + 1) / phat.shape[0]
    if target > 1:
        return 1.0

    def coverage(shift: float) -> float:
        tau = (predicted_tau - shift).clamp(0, 1)[:, None, None, None]
        return per_image_coverage(phat >= tau, labels).mean().item()

    if coverage(high) < target:
        raise ValueError("Search range does not bracket a feasible correction.")
    if coverage(low) >= target:
        return low
    for _ in range(max_iter):
        mid = (low + high) / 2
        if coverage(mid) >= target:
            high = mid
        else:
            low = mid
        if high - low <= tol:
            break
    return high
