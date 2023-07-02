"""Utility functions for BNN+VI/LV implementation."""

import inspect
from typing import Union

import torch.nn as nn

import uq_method_box.models.bnn_layers as bayesian_layers


def bnnlv_linear_layer(params, d):
    """Convert deterministic linear layer to bayesian linear layer."""
    layer = d.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)
    bnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
        bias=d.bias is not None,
        **params,
    )
    return bnn_layer


def bnnlv_conv_layer(params, d):
    """Convert deterministic convolutional layer to bayesian convolutional layer."""
    layer = d.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)  # Get BNN layer
    bnn_layer = layer_fn(
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_size=d.kernel_size,
        stride=d.stride,
        padding=d.padding,
        dilation=d.dilation,
        groups=d.groups,
        bias=d.bias is not None,
        **params,
    )
    return bnn_layer


def bnnlv_lstm_layer(params, d):
    """Convert lstm layer to bayesian lstm layer."""
    layer = d.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)  # Get BNN layer
    bnn_layer = layer_fn(
        in_features=d.input_size,
        out_features=d.hidden_size,
        bias=d.bias is not None,
        **params,
    )
    return bnn_layer


# get loss terms for energy functional
def get_log_normalizer(models: list[nn.Module]):
    """Compute terms for energy functional.

    Args:
        model: bnn with lvs model.
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
        model: bnn with lvs model.
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
        model: bnn with lvs model.
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
        new_module: nn.Module class to initialize the new layer

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


# changed partial stochastic change function to update layers to bnn+lv
def dnn_to_bnnlv_some(
    m: nn.Module,
    bnn_prior_parameters: dict[str, Union[str, float]],
    part_stoch_module_names: list[str],
) -> None:
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
    #  "More stochastic modules than modules.

    for name, value in m._modules.items():
        if m._modules[name]._modules:
            part_stoch_module_names = [
                module_name.removeprefix(name + ".")
                for module_name in part_stoch_module_names
            ]
            dnn_to_bnnlv_some(
                m._modules[name], bnn_prior_parameters, part_stoch_module_names
            )
        if name in part_stoch_module_names:
            if "Conv" in m._modules[name].__class__.__name__:
                setattr(
                    m, name, bnnlv_conv_layer(bnn_prior_parameters, m._modules[name])
                )
            elif "Linear" in m._modules[name].__class__.__name__:
                setattr(
                    m, name, bnnlv_linear_layer(bnn_prior_parameters, m._modules[name])
                )
            elif "LSTM" in m._modules[name].__class__.__name__:
                setattr(
                    m, name, bnnlv_lstm_layer(bnn_prior_parameters, m._modules[name])
                )
            else:
                pass
        else:
            continue
