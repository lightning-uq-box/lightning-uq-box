# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Public-surface inventory guards for the method/task migration."""

import importlib

import lightning_uq_box.uq_methods as uq_methods


def test_all_exported_symbols_are_bound() -> None:
    """The package export list cannot advertise a missing public symbol."""
    missing = [name for name in uq_methods.__all__ if not hasattr(uq_methods, name)]
    assert not missing


def test_canonical_and_historical_module_imports_remain_bound() -> None:
    """Canonical seams do not remove historical method submodule imports."""
    modules = (
        "base",
        "mc_dropout",
        "swag",
        "sgld",
        "masked_ensemble",
        "bnn_vi_elbo",
        "deep_ensemble",
        "deep_kernel_learning",
        "deterministic_uncertainty_estimation",
        "mean_variance_estimation",
        "deep_evidential_regression",
        "quantile_regression",
        "mixture_density",
        "density_uncertainty",
        "tasks",
        "contracts",
        "method_specs",
        "task_runtime",
    )
    for module in modules:
        imported = importlib.import_module(f"lightning_uq_box.uq_methods.{module}")
        assert imported is not None
