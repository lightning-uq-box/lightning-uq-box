# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Checked-in capability inventory for canonical UQ methods.

Only methods that have crossed the canonical task boundary appear here.  The
legacy class matrix remains intentionally absent until a method is migrated;
that prevents advertising an unsupported method/task combination.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn

from .contracts import (
    MethodSpec,
    OutputField,
    OutputSchema,
    TaskCapability,
    validate_method_specs,
)
from .tasks import (
    ClassificationTask,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
)


def _linear_smoke() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build a CPU regression smoke fixture without test-package imports."""
    return nn.Linear(2, 1), {"input": torch.zeros(2, 2), "target": torch.zeros(2, 1)}


def _classification_smoke() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build a CPU classification smoke fixture."""
    return nn.Linear(2, 3), {
        "input": torch.zeros(2, 2),
        "target": torch.zeros(2, dtype=torch.long),
    }


def _binary_smoke() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build a one-logit binary-classification smoke fixture."""
    return nn.Linear(2, 1), {"input": torch.zeros(2, 2), "target": torch.zeros(2)}


def _pixel_smoke() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build a dense prediction smoke fixture."""
    return nn.Conv2d(1, 1, kernel_size=1), {
        "input": torch.zeros(2, 1, 4, 4),
        "target": torch.zeros(2, 1, 4, 4),
    }


REGRESSION_SCHEMA = OutputSchema(
    task_type=RegressionTask,
    raw_axes=("batch", "target"),
    metric_input="raw_prediction",
    fields=(OutputField(name="pred", axes=("batch", "target")),),
)
PIXEL_REGRESSION_SCHEMA = OutputSchema(
    task_type=PixelRegressionTask,
    raw_axes=("batch", "channel", "height", "width"),
    metric_input="raw_prediction",
    fields=(OutputField(name="pred", axes=("batch", "channel", "height", "width")),),
)
CLASSIFICATION_SCHEMA = OutputSchema(
    task_type=ClassificationTask,
    modes=frozenset({"multiclass", "multilabel"}),
    raw_axes=("batch", "class"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class")),
        OutputField(name="pred_uct", axes=("batch",)),
        OutputField(name="logits", axes=("batch", "class")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)
BINARY_ONE_LOGIT_SCHEMA = OutputSchema(
    task_type=ClassificationTask,
    modes=frozenset({"binary"}),
    binary_encoding="one_logit",
    raw_axes=("batch", "class"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class")),
        OutputField(name="pred_uct", axes=("batch",)),
        OutputField(name="logits", axes=("batch", "class")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)
BINARY_TWO_LOGIT_SCHEMA = OutputSchema(
    task_type=ClassificationTask,
    modes=frozenset({"binary"}),
    binary_encoding="two_logit",
    raw_axes=("batch", "class"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class")),
        OutputField(name="pred_uct", axes=("batch",)),
        OutputField(name="logits", axes=("batch", "class")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)
SEGMENTATION_SCHEMA = OutputSchema(
    task_type=SegmentationTask,
    modes=frozenset({"multiclass", "multilabel"}),
    raw_axes=("batch", "class", "height", "width"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class", "height", "width")),
        OutputField(name="pred_uct", axes=("batch", "height", "width")),
        OutputField(name="logits", axes=("batch", "class", "height", "width")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)
SEGMENTATION_BINARY_ONE_LOGIT_SCHEMA = OutputSchema(
    task_type=SegmentationTask,
    modes=frozenset({"binary"}),
    binary_encoding="one_logit",
    raw_axes=("batch", "class", "height", "width"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class", "height", "width")),
        OutputField(name="pred_uct", axes=("batch", "height", "width")),
        OutputField(name="logits", axes=("batch", "class", "height", "width")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)
SEGMENTATION_BINARY_TWO_LOGIT_SCHEMA = OutputSchema(
    task_type=SegmentationTask,
    modes=frozenset({"binary"}),
    binary_encoding="two_logit",
    raw_axes=("batch", "class", "height", "width"),
    metric_input="logits",
    fields=(
        OutputField(name="pred", axes=("batch", "class", "height", "width")),
        OutputField(name="pred_uct", axes=("batch", "height", "width")),
        OutputField(name="logits", axes=("batch", "class", "height", "width")),
    ),
    uncertainty_fields=frozenset({"pred_uct"}),
)


DETERMINISTIC_SPEC = MethodSpec(
    name="Deterministic",
    class_path="lightning_uq_box.uq_methods.Deterministic",
    capabilities=(
        TaskCapability(
            task_type=RegressionTask,
            schema=REGRESSION_SCHEMA,
            smoke_factory_id="regression",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/regression/deterministic_canonical.yaml",
            writer="csv",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=PixelRegressionTask,
            schema=PIXEL_REGRESSION_SCHEMA,
            smoke_factory_id="pixel_regression",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/pixelwise_regression/deterministic_canonical.yaml",
            writer="hdf5",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=ClassificationTask,
            modes=frozenset({"multiclass", "multilabel"}),
            schema=CLASSIFICATION_SCHEMA,
            smoke_factory_id="classification",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/classification/deterministic_canonical.yaml",
            writer="csv",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=ClassificationTask,
            modes=frozenset({"binary"}),
            schema=BINARY_ONE_LOGIT_SCHEMA,
            smoke_factory_id="binary",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/classification/deterministic_canonical.yaml",
            writer="csv",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=ClassificationTask,
            modes=frozenset({"binary"}),
            schema=BINARY_TWO_LOGIT_SCHEMA,
            smoke_factory_id="classification",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/classification/deterministic_canonical.yaml",
            writer="csv",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=SegmentationTask,
            modes=frozenset({"multiclass", "multilabel"}),
            schema=SEGMENTATION_SCHEMA,
            smoke_factory_id="segmentation",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/image_segmentation/deterministic_canonical.yaml",
            writer="hdf5",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=SegmentationTask,
            modes=frozenset({"binary"}),
            schema=SEGMENTATION_BINARY_ONE_LOGIT_SCHEMA,
            smoke_factory_id="segmentation",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/image_segmentation/deterministic_canonical.yaml",
            writer="hdf5",
            checkpoint_fixture_id="constructed_module_strict",
        ),
        TaskCapability(
            task_type=SegmentationTask,
            modes=frozenset({"binary"}),
            schema=SEGMENTATION_BINARY_TWO_LOGIT_SCHEMA,
            smoke_factory_id="segmentation",
            test_id="tests/uq_methods/test_task_method_base.py",
            config_path="tests/configs/image_segmentation/deterministic_canonical.yaml",
            writer="hdf5",
            checkpoint_fixture_id="constructed_module_strict",
        ),
    ),
    smoke_factories={
        "regression": _linear_smoke,
        "pixel_regression": _pixel_smoke,
        "classification": _classification_smoke,
        "binary": _binary_smoke,
        "segmentation": _pixel_smoke,
    },
    lifecycle_notes="The model is attached before the eager TaskRuntime is created.",
    checkpoint_fixture_id="constructed_module_strict",
)


def _sampling_schema(schema: OutputSchema) -> OutputSchema:
    """Extend a deterministic schema with the sample axis MC Dropout exposes."""
    fields = []
    uncertainty_fields = schema.uncertainty_fields
    for output_field in schema.fields:
        axes = output_field.axes
        if output_field.name == "logits":
            axes = (*axes, "sample")
        fields.append(replace(output_field, axes=axes))
    if schema.task_type in {RegressionTask, PixelRegressionTask}:
        pred_axes = next(field.axes for field in schema.fields if field.name == "pred")
        fields.extend(
            (
                OutputField(name="pred_uct", axes=pred_axes),
                OutputField(name="epistemic_uct", axes=pred_axes),
                OutputField(name="aleatoric_uct", axes=pred_axes, required=False),
            )
        )
        uncertainty_fields = frozenset(
            {*schema.uncertainty_fields, "pred_uct", "epistemic_uct", "aleatoric_uct"}
        )
    return replace(
        schema,
        fields=tuple(fields),
        raw_axes=(*schema.raw_axes, "sample"),
        uncertainty_fields=uncertainty_fields,
    )


MCDROPOUT_SPEC = MethodSpec(
    name="MCDropout",
    class_path="lightning_uq_box.uq_methods.MCDropout",
    capabilities=tuple(
        replace(
            capability,
            schema=_sampling_schema(capability.schema),
            config_path=capability.config_path.replace(
                "deterministic_canonical", "mc_dropout_canonical"
            ),
        )
        for capability in DETERMINISTIC_SPEC.capabilities
    ),
    smoke_factories=DETERMINISTIC_SPEC.smoke_factories,
    lifecycle_notes=(
        "Dropout activation and Monte-Carlo aggregation remain method-owned; "
        "the eager runtime is created after the model is attached."
    ),
    checkpoint_fixture_id="constructed_module_strict",
)


def iter_method_specs() -> Iterator[MethodSpec]:
    """Yield every current canonical method specification."""
    yield DETERMINISTIC_SPEC
    yield MCDROPOUT_SPEC


def get_method_spec(name: str) -> MethodSpec:
    """Return a canonical method spec by public canonical name."""
    for spec in iter_method_specs():
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown canonical method: {name}")


validate_method_specs(iter_method_specs())


__all__ = [
    "BINARY_ONE_LOGIT_SCHEMA",
    "BINARY_TWO_LOGIT_SCHEMA",
    "CLASSIFICATION_SCHEMA",
    "DETERMINISTIC_SPEC",
    "MCDROPOUT_SPEC",
    "PIXEL_REGRESSION_SCHEMA",
    "REGRESSION_SCHEMA",
    "SEGMENTATION_BINARY_ONE_LOGIT_SCHEMA",
    "SEGMENTATION_BINARY_TWO_LOGIT_SCHEMA",
    "SEGMENTATION_SCHEMA",
    "get_method_spec",
    "iter_method_specs",
]
