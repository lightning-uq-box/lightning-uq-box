"""Utility functions for BNN+VI implementation."""

import torch.nn as nn

from .linear_layer import LinearFlipoutLayer

# TODO: change layers to output logs instead of kl's
# from .conv_flipout import * (these are quite a few)
# from .rnn_flipout import  LSTMFlipout
# from .linear_variational import *
# from .conv_variational import *
# from .rnn_variational import *


def bnnlv_linear_layer(params, d) -> LinearFlipoutLayer:
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


# TODO: only flipout?
# function bnnlv_conv_layer()
# function bnnlv_lstm_layer()


def bnnlv_conv_layer(params, d) -> None:
    """Convert deterministic linear layer to bayesian conv layer."""
    return NotImplementedError


def bnnlv_lstm_layer(params, d) -> None:
    """Convert deterministic linear layer to bayesian lstm layer."""
    return NotImplementedError


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
        if hasattr(layer, "log_terms"):
            if log_f_hat is None:
                log_f_hat = layer.log_terms()[0]
                log_normalizer = layer.log_terms()[1]
            else:
                log_f_hat += layer.log_terms()[0]
                log_normalizer += layer.log_terms()[1]
    return log_f_hat, log_normalizer


def get_log_Z_prior(m: nn.Module):
    """Compute summed log_Z_prior.

    Args:
        model: bnn with lvs model.
    Returns:
        summed log_Z_prior.
    """
    log_Z_prior = None
    for layer in m.modules():
        if hasattr(layer, "calc_log_Z_prior"):
            if log_Z_prior is None:
                log_Z_prior = layer.calc_log_Z_prior()
            else:
                log_Z_prior += layer.calc_log_Z_prior()
    return log_Z_prior


# changed partial stochastic change function to update layers to bnn+lv
def dnn_to_bnnlv_some(m, bnn_prior_parameters, num_stochastic_modules: int):
    """Replace linear and conv. layers with stochastic layers.

    Args:
        m: nn.module
        bnn_prior_parameter: dictionary,
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `Flipout` or `Reparameterization
        num_stochastic_modules: number of modules that should be stochastic,
            max value all modules.
    """
    # assert len(list(m.named_modules(remove_duplicate=False)))
    # >= num_stochastic_modules,
    #  "More stochastic modules than modules."

    replace_modules = list(m._modules.items())[-num_stochastic_modules:]

    for name, value in replace_modules:
        if m._modules[name]._modules:
            dnn_to_bnnlv_some(
                m._modules[name], bnn_prior_parameters, num_stochastic_modules
            )
        if "Conv" in m._modules[name].__class__.__name__:
            setattr(m, name, bnnlv_conv_layer(bnn_prior_parameters, m._modules[name]))
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(m, name, bnnlv_linear_layer(bnn_prior_parameters, m._modules[name]))
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(m, name, bnnlv_lstm_layer(bnn_prior_parameters, m._modules[name]))
        else:
            pass
    return
