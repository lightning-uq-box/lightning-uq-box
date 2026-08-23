# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Explicit contracts for the canonical method × task API.

The old concrete classes encoded these contracts implicitly in inheritance and
tensor slicing.  The canonical API makes them data so capabilities can be
validated before a trainer or a checkpoint is involved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from torch import Tensor

from .tasks import ClassificationTask, TaskSpec

if TYPE_CHECKING:
    from torch import nn


Stage = Literal["train", "validate", "test", "predict"]


@dataclass(frozen=True, kw_only=True)
class OutputField:
    """A public prediction field.

    ``axes`` describes semantic axes instead of freezing dimensions.  For
    example ``("batch", "class", "height", "width")`` remains valid for
    any image size and any number of classes.
    """

    name: str
    axes: tuple[str, ...]
    dtype: str = "floating"
    device: str = "model"
    required: bool = True
    description: str = ""


@dataclass(frozen=True, kw_only=True)
class OutputSchema:
    """Immutable output and metric contract for a method/task pair."""

    task_type: type[TaskSpec]
    modes: frozenset[str] = frozenset()
    raw_axes: tuple[str, ...]
    metric_input: str
    fields: tuple[OutputField, ...]
    binary_encoding: str | None = None
    uncertainty_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate immutable schema invariants at construction time."""
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("OutputSchema fields must have unique names.")
        if "pred" not in names:
            raise ValueError("Every OutputSchema must expose a 'pred' field.")
        if not self.metric_input:
            raise ValueError("OutputSchema.metric_input must not be empty.")
        if self.modes and not issubclass(self.task_type, ClassificationTask):
            raise ValueError("Only classification-like tasks may declare modes.")
        if self.binary_encoding and "binary" not in self.modes:
            raise ValueError("binary_encoding requires a binary schema mode.")
        if not self.uncertainty_fields.issubset(set(names)):
            raise ValueError("uncertainty_fields must be public schema fields.")

    @property
    def public_keys(self) -> frozenset[str]:
        """Return all public output keys, including optional keys."""
        return frozenset(field.name for field in self.fields)

    def accepts(self, task: TaskSpec) -> bool:
        """Whether the schema is applicable to ``task``."""
        # Capabilities are exact method × task declarations.  Segmentation is
        # a subclass of ClassificationTask for shared mode fields, but it must
        # never accidentally match a vector-classification capability.
        if type(task) is not self.task_type:
            return False
        mode = getattr(task, "mode", None)
        if self.modes and mode not in self.modes:
            return False
        if self.binary_encoding and mode == "binary":
            return getattr(task, "binary_encoding", None) == self.binary_encoding
        return True

    def validate_payload(self, payload: dict[str, Tensor]) -> None:
        """Validate public keys and the rank implied by declared axes.

        Dynamic axis dimensions and dtype families are checked by the owning
        method's tests.  This check catches the expensive-to-debug contract
        errors: a missing key, an accidental private key, or an axis squeeze
        when the batch has one sample.
        """
        known = {field.name: field for field in self.fields}
        unknown = set(payload) - set(known)
        if unknown:
            raise ValueError(
                f"Prediction payload has undeclared keys: {sorted(unknown)}"
            )
        missing = [
            field.name
            for field in self.fields
            if field.required and field.name not in payload
        ]
        if missing:
            raise ValueError(f"Prediction payload is missing required keys: {missing}")
        for name, value in payload.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Prediction payload '{name}' must be a Tensor.")
            expected_rank = len(known[name].axes)
            if value.ndim != expected_rank:
                raise ValueError(
                    f"Prediction payload '{name}' has rank {value.ndim}; "
                    f"the schema requires {expected_rank} axes {known[name].axes}."
                )


@dataclass(frozen=True, kw_only=True)
class TaskCapability:
    """A tested task/mode and output schema supported by one method."""

    task_type: type[TaskSpec]
    modes: frozenset[str] = frozenset()
    schema: OutputSchema
    stages: frozenset[Stage] = frozenset({"train", "validate", "test", "predict"})
    smoke_factory_id: str
    test_id: str
    config_path: str
    writer: str
    checkpoint_fixture_id: str
    lifecycle_notes: str = ""

    def __post_init__(self) -> None:
        """Validate that the capability exactly matches its schema."""
        if self.schema.task_type is not self.task_type:
            raise ValueError("Capability schema and task_type must agree.")
        if self.modes != self.schema.modes:
            raise ValueError("Capability modes and schema modes must agree.")
        if not self.stages:
            raise ValueError("A capability must support at least one stage.")
        if not self.smoke_factory_id:
            raise ValueError("A capability must name its smoke factory.")
        if not self.test_id or not self.config_path:
            raise ValueError("A capability must name its test and config coverage.")
        if self.writer not in {"csv", "hdf5", "none"}:
            raise ValueError("Capability writer must be 'csv', 'hdf5', or 'none'.")
        if not self.checkpoint_fixture_id:
            raise ValueError("A capability must name its checkpoint fixture.")

    def accepts(self, task: TaskSpec, stage: Stage | None = None) -> bool:
        """Return whether this capability supports ``task`` at ``stage``."""
        return self.schema.accepts(task) and (stage is None or stage in self.stages)

    @property
    def identifier(self) -> tuple[type[TaskSpec], tuple[str, ...], str | None]:
        """Return a stable key used to reject duplicate capabilities."""
        return (self.task_type, tuple(sorted(self.modes)), self.schema.binary_encoding)


SmokeFactory = Callable[[], tuple["nn.Module", dict[str, Tensor]]]


@dataclass(frozen=True, kw_only=True)
class MethodSpec:
    """Checked-in source of truth for a canonical method's capabilities."""

    name: str
    class_path: str
    capabilities: tuple[TaskCapability, ...]
    smoke_factories: dict[str, SmokeFactory] = field(default_factory=dict)
    config_paths: tuple[str, ...] = ()
    documentation_path: str | None = None
    lifecycle_notes: str = ""
    checkpoint_fixture_id: str | None = None

    def __post_init__(self) -> None:
        """Validate inventory completeness and capability uniqueness."""
        if not self.name or not self.class_path:
            raise ValueError("MethodSpec needs a name and class_path.")
        if not self.capabilities:
            raise ValueError("MethodSpec must declare at least one capability.")
        identifiers = [capability.identifier for capability in self.capabilities]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"MethodSpec {self.name} has duplicate capabilities.")
        missing_factories = {
            capability.smoke_factory_id for capability in self.capabilities
        } - set(self.smoke_factories)
        if missing_factories:
            raise ValueError(
                f"MethodSpec {self.name} is missing smoke factories: "
                f"{sorted(missing_factories)}"
            )

    def capability_for(
        self, task: TaskSpec, stage: Stage | None = None
    ) -> TaskCapability:
        """Return the one declared capability for a task/stage pair.

        Raises:
            ValueError: if the pair is unsupported or ambiguously declared.
        """
        matches = [
            capability
            for capability in self.capabilities
            if capability.accepts(task, stage)
        ]
        if len(matches) != 1:
            stage_text = "all stages" if stage is None else stage
            raise ValueError(
                f"{self.name} does not support task {task!r} for {stage_text}."
            )
        return matches[0]


def validate_method_specs(specs: Iterable[MethodSpec]) -> None:
    """Validate a registry and reject repeated canonical method names/paths."""
    specs = tuple(specs)
    names = [spec.name for spec in specs]
    paths = [spec.class_path for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("MethodSpec names must be unique.")
    if len(paths) != len(set(paths)):
        raise ValueError("MethodSpec class paths must be unique.")


__all__ = [
    "MethodSpec",
    "OutputField",
    "OutputSchema",
    "SmokeFactory",
    "Stage",
    "TaskCapability",
    "validate_method_specs",
]
