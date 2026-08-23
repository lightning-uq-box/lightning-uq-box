# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Regression tests for non-mutating prediction persistence."""

from pathlib import Path

import torch

from lightning_uq_box.uq_methods.utils import (
    save_classification_predictions,
    save_regression_predictions,
)


def test_regression_writer_does_not_mutate_samples(tmp_path: Path) -> None:
    """A writer observes a result dictionary instead of popping its samples."""
    outputs = {
        "pred": torch.zeros(1, 1),
        "samples": torch.zeros(1, 1, 2),
        "target": torch.zeros(1, 1),
    }
    save_regression_predictions(outputs, str(tmp_path / "preds.csv"))
    assert "samples" in outputs


def test_binary_writer_preserves_a_one_item_batch(tmp_path: Path) -> None:
    """One-logit binary outputs persist one CSV row rather than scalars."""
    outputs = {
        "pred": torch.tensor([[0.8]]),
        "pred_uct": torch.tensor([0.5]),
        "logits": torch.tensor([[1.4]]),
        "target": torch.tensor([1.0]),
    }
    save_classification_predictions(
        outputs, str(tmp_path / "preds.csv"), task="binary", binary_encoding="one_logit"
    )
    assert "pred" in outputs
    assert (tmp_path / "preds.csv").read_text(encoding="utf-8").count("\n") == 2
