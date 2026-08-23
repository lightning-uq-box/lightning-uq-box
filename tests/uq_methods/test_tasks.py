# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for immutable task values and their strict serialization boundary."""

import json
from typing import Any

import pytest
from torch import nn

from lightning_uq_box.uq_methods import (
    ClassificationTask,
    Deterministic,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
    task_from_mapping,
)


@pytest.mark.parametrize(
    "task",
    [
        RegressionTask(),
        ClassificationTask(mode="multiclass"),
        SegmentationTask(mode="multilabel"),
        PixelRegressionTask(),
    ],
)
def test_task_mapping_round_trip(task) -> None:
    """Every task is JSON-safe and round-trips through its allow-list."""
    serialized = json.loads(json.dumps(task.to_mapping()))
    assert task_from_mapping(serialized) == task


@pytest.mark.parametrize(
    "mapping",
    [
        {
            "version": 2,
            "class_path": "lightning_uq_box.uq_methods.tasks.RegressionTask",
            "init_args": {},
        },
        {"version": 1, "class_path": "builtins.object", "init_args": {}},
        {
            "version": 1,
            "class_path": "lightning_uq_box.uq_methods.tasks.RegressionTask",
            "init_args": {"unknown": True},
        },
    ],
)
def test_task_mapping_rejects_unrecognized_data(mapping) -> None:
    """Checkpoint task data cannot import paths or add unreviewed fields."""
    with pytest.raises(ValueError):
        task_from_mapping(mapping)


def test_task_accepts_hydra_and_lightning_cli_dialects() -> None:
    """Both supported config syntaxes construct the same task value."""
    target = "lightning_uq_box.uq_methods.tasks.ClassificationTask"
    expected = ClassificationTask(mode="multilabel")
    assert task_from_mapping({"_target_": target, "mode": "multilabel"}) == expected
    assert (
        task_from_mapping({"class_path": target, "init_args": {"mode": "multilabel"}})
        == expected
    )


def test_legacy_multilable_is_normalized_and_warned() -> None:
    """The historical spelling remains a warning-only compatibility path."""
    legacy_mode: Any = "multilable"
    with pytest.warns(DeprecationWarning):
        task = ClassificationTask(mode=legacy_mode)
    assert task.mode == "multilabel"


def test_task_saved_as_mapping_not_object() -> None:
    """Canonical hyperparameters never capture a TaskSpec object."""
    module = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    saved_task = dict(module.hparams)["task"]
    assert saved_task == RegressionTask().to_mapping()
    assert not isinstance(saved_task, RegressionTask)
