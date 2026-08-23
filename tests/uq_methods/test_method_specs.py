# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for the checked-in canonical capability inventory."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from lightning_uq_box.uq_methods import (
    ClassificationTask,
    PixelRegressionTask,
    RegressionTask,
    get_method_spec,
    iter_method_specs,
)
from lightning_uq_box.uq_methods.contracts import (
    MethodSpec,
    OutputField,
    OutputSchema,
    TaskCapability,
    validate_method_specs,
)


def _schema(**overrides) -> OutputSchema:
    """Build the smallest valid regression output schema."""
    values: dict[str, Any] = {
        "task_type": RegressionTask,
        "raw_axes": ("batch", "target"),
        "metric_input": "raw_prediction",
        "fields": (OutputField(name="pred", axes=("batch", "target")),),
    }
    values.update(overrides)
    return OutputSchema(**values)


def _capability(**overrides) -> TaskCapability:
    """Build the smallest valid regression capability."""
    values: dict[str, Any] = {
        "task_type": RegressionTask,
        "schema": _schema(),
        "smoke_factory_id": "regression",
        "test_id": "tests/uq_methods/test_method_specs.py",
        "config_path": "tests/configs/regression/deterministic_canonical.yaml",
        "writer": "csv",
        "checkpoint_fixture_id": "constructed_module_strict",
    }
    values.update(overrides)
    return TaskCapability(**values)


def _factory():
    """Supply a valid minimal factory for MethodSpec validation tests."""
    return torch.nn.Linear(1, 1), {
        "input": torch.zeros(1, 1),
        "target": torch.zeros(1, 1),
    }


def test_method_specs_are_valid_and_discoverable() -> None:
    """The registry is validated at import and supports public discovery."""
    specs = tuple(iter_method_specs())
    validate_method_specs(specs)
    assert get_method_spec("Deterministic") in specs
    assert get_method_spec("MCDropout") in specs
    for spec in specs:
        for capability in spec.capabilities:
            assert Path(capability.config_path).is_file()
            assert Path(capability.test_id).is_file()
            model, batch = spec.smoke_factories[capability.smoke_factory_id]()
            assert model is not None
            assert {"input", "target"}.issubset(batch)
    with pytest.raises(KeyError, match="Unknown canonical method"):
        get_method_spec("NotAMethod")


def test_binary_encodings_have_distinct_capabilities() -> None:
    """Binary BCE and CE are declared rather than inferred from output shape."""
    spec = get_method_spec("Deterministic")
    assert (
        spec.capability_for(
            ClassificationTask(mode="binary", binary_encoding="one_logit")
        ).schema.binary_encoding
        == "one_logit"
    )
    assert (
        spec.capability_for(
            ClassificationTask(mode="binary", binary_encoding="two_logit")
        ).schema.binary_encoding
        == "two_logit"
    )


def test_output_schema_rejects_duplicate_field_names() -> None:
    """A schema cannot give two public fields the same contract name."""
    with pytest.raises(ValueError, match="unique"):
        _schema(
            fields=(
                OutputField(name="pred", axes=("batch",)),
                OutputField(name="pred", axes=("batch",)),
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"fields": (OutputField(name="logits", axes=("batch",)),)},
        {"metric_input": ""},
        {"modes": frozenset({"binary"})},
        {
            "task_type": ClassificationTask,
            "modes": frozenset({"multiclass"}),
            "binary_encoding": "one_logit",
        },
        {"uncertainty_fields": frozenset({"pred_uct"})},
    ],
)
def test_output_schema_rejects_invalid_declarations(overrides) -> None:
    """Schemas reject invalid task/mode and public-field combinations."""
    with pytest.raises(ValueError):
        _schema(**overrides)


def test_output_schema_acceptance_and_payload_errors() -> None:
    """Schemas enforce exact task types and stable output ranks."""
    schema = _schema(
        fields=(
            OutputField(name="pred", axes=("batch", "target")),
            OutputField(name="optional", axes=("batch",), required=False),
        )
    )
    assert schema.public_keys == frozenset({"pred", "optional"})
    assert schema.accepts(RegressionTask())
    assert not schema.accepts(PixelRegressionTask())
    with pytest.raises(ValueError, match="undeclared"):
        schema.validate_payload({"pred": torch.zeros(1, 1), "unknown": torch.zeros(1)})
    with pytest.raises(ValueError, match="missing"):
        schema.validate_payload({})
    with pytest.raises(TypeError, match="Tensor"):
        schema.validate_payload(cast(dict[str, torch.Tensor], {"pred": "not-a-tensor"}))
    with pytest.raises(ValueError, match="rank"):
        schema.validate_payload({"pred": torch.zeros(1)})


@pytest.mark.parametrize(
    "overrides",
    [
        {"task_type": ClassificationTask},
        {"modes": frozenset({"binary"})},
        {"stages": frozenset()},
        {"smoke_factory_id": ""},
        {"test_id": ""},
        {"config_path": ""},
        {"writer": "parquet"},
        {"checkpoint_fixture_id": ""},
    ],
)
def test_task_capability_rejects_incomplete_contracts(overrides) -> None:
    """Capabilities require one complete, matching vertical-slice contract."""
    with pytest.raises(ValueError):
        _capability(**overrides)


def test_method_spec_and_registry_error_paths() -> None:
    """The capability registry rejects incomplete and ambiguous inventories."""
    capability = _capability()
    with pytest.raises(ValueError):
        MethodSpec(name="", class_path="example.Method", capabilities=(capability,))
    with pytest.raises(ValueError):
        MethodSpec(name="Example", class_path="example.Method", capabilities=())
    with pytest.raises(ValueError):
        MethodSpec(
            name="Example",
            class_path="example.Method",
            capabilities=(capability, capability),
            smoke_factories={"regression": _factory},
        )
    with pytest.raises(ValueError):
        MethodSpec(
            name="Example", class_path="example.Method", capabilities=(capability,)
        )

    spec = MethodSpec(
        name="Example",
        class_path="example.Method",
        capabilities=(capability,),
        smoke_factories={"regression": _factory},
    )
    assert spec.capability_for(RegressionTask()).accepts(RegressionTask(), "test")
    with pytest.raises(ValueError, match="does not support"):
        spec.capability_for(ClassificationTask(mode="multiclass"), "predict")
    with pytest.raises(ValueError, match="names must be unique"):
        validate_method_specs((spec, spec))
    duplicate_path = MethodSpec(
        name="Other",
        class_path="example.Method",
        capabilities=(capability,),
        smoke_factories={"regression": _factory},
    )
    with pytest.raises(ValueError, match="class paths must be unique"):
        validate_method_specs((spec, duplicate_path))
