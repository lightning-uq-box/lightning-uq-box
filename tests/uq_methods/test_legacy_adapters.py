# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""0.4 compatibility checks for migrated concrete task class adapters."""

import importlib
import inspect

import pytest
from torch import nn

from lightning_uq_box.uq_methods import DeterministicRegression, MCDropoutRegression


def test_deterministic_regression_adapter_warns_and_keeps_signature() -> None:
    """The legacy deterministic name remains callable with its old arguments."""
    parameters = tuple(inspect.signature(DeterministicRegression).parameters)
    assert parameters == (
        "model",
        "loss_fn",
        "freeze_backbone",
        "optimizer",
        "lr_scheduler",
    )
    with pytest.warns(DeprecationWarning, match="DeterministicRegression"):
        module = DeterministicRegression(nn.Linear(2, 1), nn.MSELoss())
    assert all(key.startswith("model.") for key in module.state_dict())


def test_mc_dropout_adapter_warns_and_historical_module_imports() -> None:
    """MC Dropout keeps both its local import path and state-key topology."""
    module_path = importlib.import_module("lightning_uq_box.uq_methods.mc_dropout")
    assert module_path.MCDropoutRegression is MCDropoutRegression
    with pytest.warns(DeprecationWarning, match="MCDropoutRegression"):
        module = MCDropoutRegression(
            nn.Sequential(nn.Linear(2, 2), nn.Dropout(), nn.Linear(2, 1)),
            num_mc_samples=2,
            loss_fn=nn.MSELoss(),
        )
    assert all(key.startswith("model.") for key in module.state_dict())
