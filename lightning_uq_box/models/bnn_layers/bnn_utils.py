# BSD 3-Clause License

# Copyright (c) 2023, Intel Labs
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utility functions for BNN Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to be trained with the Energy Loss and support batched inputs.
"""

import math
from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor

import lightning_uq_box.models.bnn_layers as bayesian_layers


def bnn_linear_layer(params: dict[str, Any], linear_layer: nn.Linear) -> nn.Module:
    """Convert deterministic linear layer to bayesian linear layer."""
    layer = linear_layer.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)
    bnn_layer = layer_fn(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        **params,
    )
    return bnn_layer


def bnn_conv_layer(
    params: dict[str, Any], conv_layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
) -> nn.Module:
    """Convert deterministic convolutional layer to bayesian convolutional layer."""
    layer = conv_layer.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)
    bnn_layer = layer_fn(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=conv_layer.bias is not None,
        **params,
    )
    return bnn_layer


def bnn_lstm_layer(params: dict[str, Any], lstm_layer: nn.LSTM) -> nn.Module:
    """Convert lstm layer to bayesian lstm layer."""
    layer = lstm_layer.__class__.__name__ + "Variational"
    layer_fn = getattr(bayesian_layers, layer)
    bnn_layer = layer_fn(
        in_features=lstm_layer.input_size,
        out_features=lstm_layer.hidden_size,
        bias=lstm_layer.bias is not None,
        **params,
    )
    return bnn_layer


def convert_deterministic_to_bnn(
    deterministic_model: nn.Module,
    bnn_prior_parameters: dict[str, Any],
    stochastic_module_names: list[str],
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
            bayesian_layer_type: `flipout` or `reparameterization
        stochastic_module_names: list of module names that should become stochastic
    """
    for name in stochastic_module_names:
        layer_names = name.split(".")
        current_module = deterministic_model
        for l_name in layer_names[:-1]:
            current_module = dict(current_module.named_children())[l_name]

        target_layer_name = layer_names[-1]
        current_layer = dict(current_module.named_children())[target_layer_name]

        if "Conv" in current_layer.__class__.__name__:
            setattr(
                current_module,
                target_layer_name,
                bnn_conv_layer(bnn_prior_parameters, current_layer),
            )
        elif "Linear" in current_layer.__class__.__name__:
            setattr(
                current_module,
                target_layer_name,
                bnn_linear_layer(bnn_prior_parameters, current_layer),
            )
        elif "LSTM" in current_layer.__class__.__name__:
            setattr(
                current_module,
                target_layer_name,
                bnn_lstm_layer(bnn_prior_parameters, current_layer),
            )
        else:
            pass


def get_kl_loss(model: nn.Module) -> Tensor:
    """Compute the KL Loss of model.

    Args:
        model: pytorch model with variational layers

    Returns:
        summed kl loss of model
    """
    kl_loss = None
    for layer in model.modules():
        if hasattr(layer, "kl_loss"):
            if kl_loss is None:
                kl_loss = layer.kl_loss()
            else:
                kl_loss += layer.kl_loss()
    return kl_loss


def calc_log_f_hat(w: Tensor, m_W: Tensor, std_W: Tensor, prior_sigma: float) -> Tensor:
    """Compute single summand in equation 3.16.

    Args:
        w: weight matrix [num_params]
        m_W: mean weight matrix at current iteration [num_params]
        std_W: sigma weight matrix at current iteration [num_params]
        prior_sigma: Prior variance of weights

    Returns:
        log f hat summed over the parameters shape 0
    """
    v_W = std_W**2
    # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
    # \lambda is (\lambda_q - \lambda_prior) / N
    # assuming prior mean is 0 and moving N calculation outside
    out = ((v_W - prior_sigma) / (2 * prior_sigma * v_W)) * (w**2) + (m_W / v_W) * w
    # sum out over all dimension except first

    # this does not support both linear anc conv layers
    # conv layers have number of filters in first dimension
    # return torch.sum(out, dim=tuple(range(1, out.ndim)))
    return torch.atleast_1d(torch.sum(out))  # keep dimension 0


def calc_log_normalizer(m_W: Tensor, std_W: Tensor) -> Tensor:
    """Compute single left summand of 3.18.

    Args:
        m_W: mean weight matrix at current iteration [num_params]
        std_W: sigma weight matrix at current iteration [num_params]

    Returns:
        log normalizer summed over all layer parameters shape 0
    """
    v_W = std_W**2
    return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
