# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Serializable task descriptions used by canonical UQ methods.

Task descriptions deliberately contain *only* semantic task information.  They
are not ``nn.Module`` instances and are never saved in a checkpoint as Python
objects.  :func:`normalize_task` accepts the two configuration dialects used by
Lightning-UQ-Box and turns them into one of the immutable values below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Literal
from warnings import warn

TaskMode = Literal["binary", "multiclass", "multilabel"]
BinaryEncoding = Literal["one_logit", "two_logit"]

_TASK_VERSION = 1
_TASK_MODULE = "lightning_uq_box.uq_methods.tasks"


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    """Base class for a task value.

    Subclasses define the public fields that may be represented in a config or
    checkpoint.  The implementation intentionally uses an allow-list instead
    of importing arbitrary class paths from untrusted checkpoint data.
    """

    class_path: ClassVar[str]

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable, JSON-serializable checkpoint representation."""
        return {
            "version": _TASK_VERSION,
            "class_path": self.class_path,
            "init_args": asdict(self),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskSpec:
        """Construct an allow-listed task from a config/checkpoint mapping.

        Args:
            value: a versioned serialized task, a LightningCLI
                ``class_path``/``init_args`` mapping, or a Hydra
                ``_target_`` mapping.

        Returns:
            the normalized immutable task value.

        Raises:
            TypeError: if ``value`` is not a supported task mapping.
            ValueError: if the mapping has unknown or missing fields.
        """
        return task_from_mapping(value)


@dataclass(frozen=True, kw_only=True)
class RegressionTask(TaskSpec):
    """A vector or scalar regression task."""

    class_path: ClassVar[str] = f"{_TASK_MODULE}.RegressionTask"


@dataclass(frozen=True, kw_only=True)
class ClassificationTask(TaskSpec):
    """A classification task with an explicit probability encoding.

    ``binary_encoding`` matters only for the binary mode.  Keeping it in the
    value prevents an output conversion from guessing a distribution based on
    a last-axis size.  A one-logit binary task uses BCE/sigmoid; a two-logit
    binary task uses CE/softmax.
    """

    mode: TaskMode = "multiclass"
    binary_encoding: BinaryEncoding = "one_logit"

    class_path: ClassVar[str] = f"{_TASK_MODULE}.ClassificationTask"

    def __post_init__(self) -> None:
        """Validate modes and normalize the historical misspelling."""
        mode = self.mode
        if mode == "multilable":  # type: ignore[comparison-overlap]
            warn(
                "'multilable' is deprecated; use 'multilabel'.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "mode", "multilabel")
            mode = "multilabel"
        if mode not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(
                "ClassificationTask.mode must be 'binary', 'multiclass', or "
                f"'multilabel', got {mode!r}."
            )
        if self.binary_encoding not in {"one_logit", "two_logit"}:
            raise ValueError(
                "ClassificationTask.binary_encoding must be 'one_logit' or "
                f"'two_logit', got {self.binary_encoding!r}."
            )


@dataclass(frozen=True, kw_only=True)
class SegmentationTask(ClassificationTask):
    """A dense classification task with class axis one."""

    class_path: ClassVar[str] = f"{_TASK_MODULE}.SegmentationTask"


@dataclass(frozen=True, kw_only=True)
class PixelRegressionTask(TaskSpec):
    """A dense regression task with channel axis one."""

    class_path: ClassVar[str] = f"{_TASK_MODULE}.PixelRegressionTask"


_TASK_TYPES: dict[str, type[TaskSpec]] = {
    RegressionTask.class_path: RegressionTask,
    ClassificationTask.class_path: ClassificationTask,
    SegmentationTask.class_path: SegmentationTask,
    PixelRegressionTask.class_path: PixelRegressionTask,
    # The package-level exports are supported config entry points.  They map
    # to the same fixed allow-list and serialize back to the canonical module
    # path above, so accepting them does not permit arbitrary imports.
    "lightning_uq_box.uq_methods.RegressionTask": RegressionTask,
    "lightning_uq_box.uq_methods.ClassificationTask": ClassificationTask,
    "lightning_uq_box.uq_methods.SegmentationTask": SegmentationTask,
    "lightning_uq_box.uq_methods.PixelRegressionTask": PixelRegressionTask,
}


def _task_fields(task_type: type[TaskSpec]) -> set[str]:
    return {field.name for field in fields(task_type) if field.init}


def task_from_mapping(value: Mapping[str, Any]) -> TaskSpec:
    """Deserialize an allow-listed task mapping.

    This is intentionally strict.  In particular it rejects a checkpoint that
    claims to be a newer representation or names a Python path outside of this
    module rather than importing that path dynamically.
    """
    raw = dict(value)
    if "_target_" in raw:
        allowed = {"_target_", *(_task_fields_from_raw_target(raw))}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown task fields: {sorted(unknown)}")
        class_path = raw.pop("_target_")
        init_args = raw
    elif "class_path" in raw and "init_args" in raw:
        allowed = {"class_path", "init_args", "version"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown task mapping fields: {sorted(unknown)}")
        version = raw.get("version")
        if version is not None and version != _TASK_VERSION:
            raise ValueError(
                f"Unsupported task serialization version {version!r}; "
                f"expected {_TASK_VERSION}."
            )
        class_path = raw["class_path"]
        init_args = raw["init_args"]
    else:
        raise TypeError(
            "A task mapping must use Hydra '_target_' or LightningCLI "
            "'class_path' and 'init_args' keys."
        )

    if not isinstance(class_path, str) or class_path not in _TASK_TYPES:
        raise ValueError(f"Unsupported task class_path: {class_path!r}.")
    if not isinstance(init_args, Mapping):
        raise TypeError("task.init_args must be a mapping.")

    task_type = _TASK_TYPES[class_path]
    args = dict(init_args)
    unknown_args = set(args) - _task_fields(task_type)
    if unknown_args:
        raise ValueError(f"Unknown task init_args: {sorted(unknown_args)}")
    return task_type(**args)


def _task_fields_from_raw_target(value: Mapping[str, Any]) -> set[str]:
    """Return accepted Hydra task fields without importing a target."""
    class_path = value.get("_target_")
    if not isinstance(class_path, str) or class_path not in _TASK_TYPES:
        raise ValueError(f"Unsupported task class_path: {class_path!r}.")
    return _task_fields(_TASK_TYPES[class_path])


def normalize_task(
    task: TaskSpec | Mapping[str, Any] | None, *, default: TaskSpec | None = None
) -> TaskSpec | None:
    """Normalize a public task argument without providing a mutable default.

    Args:
        task: a task value, supported config mapping, or ``None``.
        default: a freshly constructed default supplied by the owning method.

    Returns:
        the supplied task, its deserialized mapping, or ``default``.
    """
    if task is None:
        return default
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Mapping):
        return task_from_mapping(task)
    raise TypeError("task must be a TaskSpec, supported mapping, or None.")


__all__ = [
    "BinaryEncoding",
    "ClassificationTask",
    "PixelRegressionTask",
    "RegressionTask",
    "SegmentationTask",
    "TaskMode",
    "TaskSpec",
    "normalize_task",
    "task_from_mapping",
]
