# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test that Lightning hooks are declared with Lightning's own signature."""

import inspect

import pytest
from lightning import LightningModule

import lightning_uq_box.uq_methods as uq_methods

HOOKS = ["on_test_batch_end"]


def _classes_defining(hook: str) -> list[type]:
    """Exported UQ methods that define their own version of ``hook``.

    Args:
        hook: name of the Lightning hook

    Returns:
        list of classes with ``hook`` in their own ``__dict__``
    """
    return [
        obj
        for name in uq_methods.__all__
        if isinstance(obj := getattr(uq_methods, name), type)
        and issubclass(obj, LightningModule)
        and hook in obj.__dict__
    ]


@pytest.mark.parametrize("hook", HOOKS)
def test_hook_is_defined_by_some_class(hook: str) -> None:
    assert _classes_defining(hook), f"no exported class overrides {hook}"


@pytest.mark.parametrize(
    "hook,cls",
    [(hook, cls) for hook in HOOKS for cls in _classes_defining(hook)],
    ids=lambda p: p if isinstance(p, str) else p.__name__,
)
def test_hook_signature_matches_lightning(hook: str, cls: type) -> None:
    expected = list(inspect.signature(getattr(LightningModule, hook)).parameters)
    actual = list(inspect.signature(cls.__dict__[hook]).parameters)
    assert actual == expected, (
        f"{cls.__name__}.{hook} declares {actual}, but Lightning calls it "
        f"positionally with {expected}"
    )
