# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for BNN+VI/LV implementation."""

import inspect
from typing import Union

import torch.nn as nn


# get loss terms for energy functional
def get_log_normalizer(models: list[nn.Module]):
    """Compute terms for energy functional.

    Args:
        models: list of models like the bnn and lvs model
            to retrieve the log normalizer from

    Returns:
        log_normalizer: (3.18) in [1].
        [1]: Depeweg, Stefan. Modeling epistemic and aleatoric uncertainty
        with Bayesian neural networks and latent variables.
        Diss. Technische Universität München, 2019.
    """
    log_normalizer = None
    for m in models:
        for layer in m.modules():
            if hasattr(layer, "log_normalizer"):
                if log_normalizer is None:
                    log_normalizer = layer.log_normalizer()
                else:
                    log_normalizer += layer.log_normalizer()
    return log_normalizer


def get_log_f_hat(models: list[nn.Module]):
    """Compute summed log_f_hat.

    Args:
        models: list of models like the bnn and lvs model
            to retrieve the log_f_hat term from

    Returns:
        log_f_hat: log of (3.16) in [1].
        [1]: Depeweg, Stefan. Modeling epistemic and aleatoric uncertainty
        with Bayesian neural networks and latent variables.
        Diss. Technische Universität München, 2019.
    """
    log_f_hat = None
    for m in models:
        for layer in m.modules():
            if hasattr(layer, "log_f_hat"):
                if log_f_hat is None:
                    log_f_hat = layer.log_f_hat()
                else:
                    log_f_hat += layer.log_f_hat()
    return log_f_hat


def get_log_Z_prior(models: list[nn.Module]):
    """Compute summed log_Z_prior.

    Args:
        models: list of models like the bnn and lvs model
            to retrieve the summed log_Z_prior from

    Returns:
        summed log_Z_prior.
    """
    log_Z_prior = None
    for m in models:
        for layer in m.modules():
            if hasattr(layer, "calc_log_Z_prior"):
                if log_Z_prior is None:
                    log_Z_prior = layer.calc_log_Z_prior()
                else:
                    log_Z_prior += layer.calc_log_Z_prior()
    return log_Z_prior


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by name.

    Args:
        model: full model
        module_name: name of module to replace within model
        new_module: initialized module which is the replacement
    """
    module_levels = module_name.split(".")
    last_level = module_levels[-1]
    if len(module_levels) == 1:
        setattr(model, last_level, new_module)
    else:
        setattr(getattr(model, ".".join(module_levels[:-1])), last_level, new_module)


def retrieve_module_init_args(
    current_module: type[nn.Module],
) -> dict[str, Union[str, float, int, bool]]:
    """Reinitialize a new layer with arguments.

    Args:
        current_module: nn.Module class to initialize the new layer

    Returns:
        nn.Modules init args
    """
    current_init_arg_names = list(inspect.signature(current_module.__init__).parameters)
    current_args: dict[str, Union[str, float, int, bool]] = {}
    for name in current_init_arg_names:
        if name == "bias":
            current_args[name] = current_module.bias is not None
            continue
        try:
            current_args[name] = getattr(current_module, name)
        except AttributeError:
            # some init args are not necessarily defined by default
            # like device, dtype
            continue
    return current_args
