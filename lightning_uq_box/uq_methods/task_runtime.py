# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Private runtime services for canonical task-aware methods.

``TaskRuntime`` is deliberately a child module so Lightning moves it with its
parent and hook ownership is unambiguous.  Metric collections are deliberately
kept out of the module registry: metric state belongs to a run, never to a
model checkpoint.  The runtime itself therefore contributes no persistent
state-dict keys.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import MetricCollection

from .contracts import OutputSchema
from .tasks import (
    ClassificationTask,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
    TaskSpec,
)
from .utils import (
    default_classification_metrics,
    default_px_regression_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    save_classification_predictions,
    save_image_predictions,
    save_regression_predictions,
)


class TaskRuntime(nn.Module):
    """Run-only task handling, metrics, result shaping, and persistence."""

    def __init__(self, task: TaskSpec, schema: OutputSchema, num_outputs: int) -> None:
        """Initialize the eager task runtime.

        Args:
            task: normalized task semantics.
            schema: selected method/task output contract.
            num_outputs: number of model output channels/classes.
        """
        super().__init__()
        self.task = task
        self.schema = schema
        self.num_outputs = num_outputs
        # Assign outside nn.Module.__setattr__: torchmetrics state must not
        # become task_runtime.* checkpoint keys.
        object.__setattr__(
            self,
            "_metrics",
            {
                "train": self._make_metrics("train"),
                "validate": self._make_metrics("val"),
                "test": self._make_metrics("test"),
            },
        )

    def _make_metrics(self, prefix: str) -> MetricCollection:
        """Create task-appropriate metrics for one lifecycle stage."""
        if isinstance(self.task, PixelRegressionTask):
            return default_px_regression_metrics(prefix)
        if isinstance(self.task, RegressionTask):
            return default_regression_metrics(prefix, include_r2=False)
        if isinstance(self.task, SegmentationTask):
            return default_segmentation_metrics(
                prefix, self.task.mode, self.num_outputs
            )
        if isinstance(self.task, ClassificationTask):
            return default_classification_metrics(
                prefix, self.task.mode, self.num_outputs
            )
        raise TypeError(f"Unsupported task runtime: {self.task!r}")

    @property
    def train_metrics(self) -> MetricCollection:
        """Metrics accumulated while training."""
        return self._metrics["train"]

    @property
    def val_metrics(self) -> MetricCollection:
        """Metrics accumulated while validating."""
        return self._metrics["validate"]

    @property
    def test_metrics(self) -> MetricCollection:
        """Metrics accumulated while testing."""
        return self._metrics["test"]

    def normalize_target(self, target: Tensor) -> Tensor:
        """Normalize only an explicit singleton class axis.

        Batch dimensions are never squeezed.  This is the important distinction
        from the legacy ``squeeze(-1)`` behavior: a one-item batch remains a
        batch, and a spatial width of one is not accidentally removed.
        """
        if isinstance(self.task, ClassificationTask) and self.task.mode == "multiclass":
            if target.ndim >= 2 and target.shape[1] == 1:
                return target.select(1, 0)
        if (
            isinstance(self.task, ClassificationTask)
            and self.task.mode == "binary"
            and self.task.binary_encoding == "one_logit"
            and target.ndim >= 2
            and target.shape[1] == 1
        ):
            return target.select(1, 0)
        return target

    def target_for_loss(self, target: Tensor, raw_output: Tensor) -> Tensor:
        """Return the target shape required by the explicit task loss contract.

        BCE with one-logit outputs is the one case where metric targets and
        loss targets intentionally differ: metrics consume ``[batch, ...]``
        after removing the class axis, while BCE consumes the matching
        ``[batch, 1, ...]`` shape.  The axis insertion is driven by the task's
        declared encoding, never by a generic final-axis squeeze.
        """
        if (
            isinstance(self.task, ClassificationTask)
            and self.task.mode == "binary"
            and self.task.binary_encoding == "one_logit"
            and target.ndim == raw_output.ndim - 1
        ):
            return target.unsqueeze(1)
        if isinstance(self.task, ClassificationTask) and self.task.mode == "multiclass":
            return self.normalize_target(target)
        return target

    def update_metrics(self, stage: str, prediction: Tensor, target: Tensor) -> None:
        """Update metrics for a batch, including batches of size one."""
        metrics = self._metrics[stage]
        # The collections are intentionally not registered child modules, so
        # explicitly move their non-persistent state before their first update.
        metrics.to(prediction.device)
        metrics.update(prediction, self.normalize_target(target))

    def compute_and_reset(self, stage: str) -> dict[str, Tensor]:
        """Compute and reset metrics for an epoch boundary."""
        metrics = self._metrics[stage]
        computed = metrics.compute()
        metrics.reset()
        return computed

    def test_result(
        self,
        payload: Mapping[str, Tensor],
        batch: Mapping[str, Any],
        *,
        input_key: str,
        target_key: str,
    ) -> dict[str, Any]:
        """Copy a prediction payload and add a normalized target/auxiliary data."""
        result: dict[str, Any] = {
            key: value.detach().cpu().clone() for key, value in payload.items()
        }
        target = batch[target_key]
        if not isinstance(target, Tensor):
            raise TypeError(f"Batch target '{target_key}' must be a Tensor.")
        result[target_key] = self.normalize_target(target).detach().cpu().clone()
        for key, value in batch.items():
            if key in {input_key, target_key}:
                continue
            if isinstance(value, Tensor):
                result[key] = value.detach().cpu().clone()
            else:
                result[key] = value
        return result

    def on_test_start(self, root_dir: str, save_predictions: bool) -> str | None:
        """Create the dense-prediction directory when required."""
        if not save_predictions or not isinstance(
            self.task, SegmentationTask | PixelRegressionTask
        ):
            return None
        path = os.path.join(root_dir, "preds")
        os.makedirs(path, exist_ok=True)
        return path

    def write_test_result(
        self,
        outputs: STEP_OUTPUT,
        *,
        root_dir: str,
        batch_idx: int,
        save_predictions: bool,
        prediction_dir: str | None,
    ) -> None:
        """Persist a copied test result according to the task contract."""
        if not isinstance(outputs, dict):
            return
        copied = {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in outputs.items()
        }
        if isinstance(self.task, SegmentationTask | PixelRegressionTask):
            if save_predictions and prediction_dir is not None:
                save_image_predictions(copied, batch_idx, prediction_dir)
            return
        if isinstance(self.task, RegressionTask):
            save_regression_predictions(copied, os.path.join(root_dir, "preds.csv"))
            return
        if isinstance(self.task, ClassificationTask):
            save_classification_predictions(
                copied,
                os.path.join(root_dir, "preds.csv"),
                task=self.task.mode,
                binary_encoding=self.task.binary_encoding,
            )


__all__ = ["TaskRuntime"]
