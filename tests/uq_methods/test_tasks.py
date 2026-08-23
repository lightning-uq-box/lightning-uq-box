# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for immutable task values and their strict serialization boundary."""

import json
from typing import Any, cast

import pytest
from torch import nn

from lightning_uq_box.uq_methods import (
    ClassificationTask,
    Deterministic,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
    normalize_task,
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


def test_task_classmethod_and_normalization_cover_public_entry_points() -> None:
    """Task values, supported mappings, and explicit defaults normalize safely."""
    mapping = RegressionTask().to_mapping()
    assert RegressionTask.from_mapping(mapping) == RegressionTask()
    assert normalize_task(RegressionTask()) == RegressionTask()
    assert normalize_task(mapping) == RegressionTask()
    assert normalize_task(None, default=PixelRegressionTask()) == PixelRegressionTask()


@pytest.mark.parametrize(
    "mode, binary_encoding",
    [("not-a-mode", "one_logit"), ("binary", "not-an-encoding")],
)
def test_classification_task_rejects_invalid_semantics(
    mode: Any, binary_encoding: Any
) -> None:
    """Invalid modes and probability encodings fail before model construction."""
    with pytest.raises(ValueError):
        ClassificationTask(mode=mode, binary_encoding=binary_encoding)


@pytest.mark.parametrize(
    "mapping, exception",
    [
        (
            {
                "_target_": "lightning_uq_box.uq_methods.tasks.RegressionTask",
                "unexpected": True,
            },
            ValueError,
        ),
        (
            {
                "class_path": "lightning_uq_box.uq_methods.tasks.RegressionTask",
                "init_args": {},
                "unexpected": True,
            },
            ValueError,
        ),
        ({"not_a_task": True}, TypeError),
        (
            {
                "class_path": "lightning_uq_box.uq_methods.tasks.RegressionTask",
                "init_args": [],
            },
            TypeError,
        ),
        ({"_target_": "builtins.object"}, ValueError),
    ],
)
def test_task_mapping_rejects_every_untrusted_shape(
    mapping: dict[str, Any], exception
) -> None:
    """The task deserializer has no dynamic import or ignored-field fallback."""
    with pytest.raises(exception):
        task_from_mapping(mapping)


def test_normalize_task_rejects_non_task_values() -> None:
    """Only task values, allow-listed mappings, and ``None`` are accepted."""
    with pytest.raises(TypeError):
        normalize_task(cast(Any, "regression"))
