"""Utility functions for BNN+VI implementation."""

from typing import Any

import torch.nn as nn

from .linear_layer import LinearFlipoutLayer


def our_bnn_linear_layer(params, d) -> LinearFlipoutLayer:
    """Convert deterministic linear layer to bayesian linear layer."""
    bnn_layer = LinearFlipoutLayer(
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


# get loss terms for energy functional


def get_log_f_log_normalizer(m: nn.Module):
    """Compute terms for energy functional.

    Args:
        model: bnn with lvs model.
    Returns:
        log_f_hat: log of (3.16) in [1].
        log_normalizer: (3.18) in [1].
        [1]: Depeweg, Stefan. Modeling epistemic and aleatoric uncertainty
        with Bayesian neural networks and latent variables.
        Diss. Technische Universität München, 2019.
    """
    log_f_hat = None
    log_normalizer = None
    for layer in m.modules():
        if hasattr(layer, "log_f_hat"):
            if log_f_hat is None:
                log_f_hat = layer.log_f_hat()
                log_normalizer = layer.log_normalizer()
            else:
                log_f_hat += layer.log_f_hat()
                log_normalizer += layer.log_normalizer()
    return log_f_hat, log_normalizer
