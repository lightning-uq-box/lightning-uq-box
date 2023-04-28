"""Utility functions for BNN+VI implementation."""

from typing import Any

import torch.nn as nn

from .linear_layer import LinearReparameterizationLayer


def our_bnn_linear_layer(params, d) -> LinearReparameterizationLayer:
    """Convert deterministic linear layer to bayesian linear layer."""
    bnn_layer = LinearReparameterizationLayer(
        in_features=d.in_features,
        out_features=d.out_features,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )
    return bnn_layer


def linear_dnn_to_bnn(model: nn.Module, bnn_prior_parameters: dict[str, Any]) -> None:
    """Convert linear MLP to bayesian MLP.

    Args:
        model: mlp model to convert
        bnn_prior_parameters: dictionary with parameters for bnn
            layer initialization
    """
    for name, value in list(model._modules.items()):
        if model._modules[name]._modules:
            linear_dnn_to_bnn(model._modules[name], bnn_prior_parameters)
        elif "Linear" in model._modules[name].__class__.__name__:
            setattr(
                model,
                name,
                our_bnn_linear_layer(bnn_prior_parameters, model._modules[name]),
            )
    return
