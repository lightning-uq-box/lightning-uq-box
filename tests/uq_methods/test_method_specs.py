# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for the checked-in canonical capability inventory."""

from pathlib import Path

from lightning_uq_box.uq_methods import (
    ClassificationTask,
    get_method_spec,
    iter_method_specs,
)
from lightning_uq_box.uq_methods.contracts import validate_method_specs


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
