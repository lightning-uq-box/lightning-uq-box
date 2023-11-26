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

"""Base Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to be trained with the Energy Loss and support batched inputs.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from lightning_uq_box.models.bnn_layers.bnn_utils import (
    calc_log_f_hat,
    calc_log_normalizer,
)


class BaseVariationalLayer_(nn.Module):
    """Base Variational Layer for BNN Layers."""

    valid_layer_types = ["reparameterization", "flipout"]

    def __init__(
        self,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ) -> None:
        """Initialize a new instance of Base Variational Layer.

        Args:
            prior_mu: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_sigma: variance of the prior arbitrary
                distribution to be used on the complexity cost,
            posterior_mu_init: init trainable mu parameter
                representing mean of the approximate posterior,
            posterior_rho_init: init trainable rho parameter
                representing the sigma of the approximate
                posterior through softplus function,
            bias: if set to False, the layer will not learn an additive bias.
            layer_type: reparameterization trick with
                "reparameterization" or "flipout".
        """
        super().__init__()

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        assert (
            layer_type in self.valid_layer_types
        ), f"Only {self.valid_layer_types} are valid layer types but found {layer_type}"
        self.layer_type = layer_type
        self.is_frozen = False

    def define_bayesian_parameters(self):
        """Define Bayesian parameters."""
        raise NotImplementedError

    def init_parameters(self):
        """Initialize Bayesian Parameters."""
        self.prior_weight_mu.data.fill_(self.prior_mu)
        self.prior_weight_sigma.fill_(self.prior_sigma)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.0)
        if self.bias:
            self.prior_bias_mu.data.fill_(self.prior_mu)
            self.prior_bias_sigma.fill_(self.prior_sigma)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.0)

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_weight.numel()
        if self.bias:
            n_params += self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_sigma**2 * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Returns:
            log_normalizer.
        """
        # compute variance of weight from unconstrained variable rho_weight
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_weight, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Returns:
            log_f_hat.
        """
        # compute variance of weight from unconstrained variable rho_weight
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = sigma_weight * self.eps_weight

        # sampling weight
        weight = self.mu_weight + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_weight,
            std_W=sigma_weight,
            prior_sigma=self.prior_sigma,
        )

        bias = None
        # get log_f_hat for biases
        # first sample bias
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias
            bias = self.mu_bias + delta_bias
            # compute log_f_hat for weights and biases
            log_f_hat = log_f_hat + calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias, prior_sigma=self.prior_sigma
            )

        return log_f_hat

    def freeze_layer(self) -> None:
        """Freeze Variational Layers.

        This is useful when using BNN+LV to fix the BNN parameters
        to sample the Latent Variables to estimate aleatoric uncertainy.
        """
        self.is_frozen = True

    def unfreeze_layer(self) -> None:
        """Unfreeze Variational Layers."""
        self.is_frozen = False

    def kl_div(
        self, mu_q: Tensor, sigma_q: Tensor, mu_p: Tensor, sigma_p: Tensor
    ) -> Tensor:
        """Compute kl divergence between two gaussians (Q || P).

        Args:
            mu_q: mu parameter of distribution Q
            sigma_q: sigma parameter of distribution Q
            mu_p: mu parameter of distribution P
            sigma_p: sigma parameter of distribution P

        Returns:
            kl divergence
        """
        kl = (
            torch.log(sigma_p)
            - torch.log(sigma_q)
            + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2 * (sigma_p**2))
            - 0.5
        )
        return kl.mean()

    def kl_loss(self) -> Tensor:
        """Compute the KL Loss of the layer."""
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma
        )
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma
            )
        return kl


class BaseConvLayer_(BaseVariationalLayer_):
    """Base Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -3,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ) -> None:
        """Initialize a new instance of BaseConvLayer.

        Args:
            prior_mu: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_sigma: variance of the prior arbitrary
                distribution to be used on the complexity cost,
            posterior_mu_init: init trainable mu parameter
                representing mean of the approximate posterior,
            posterior_rho_init: init trainable rho parameter
                representing the sigma of the approximate
                posterior through softplus function,
            bias: if set to False, the layer will not learn an additive bias.
            layer_type: reparameterization trick with
                "reparameterization" or "flipout".
        """
        super().__init__(
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bias,
            layer_type,
        )

        if in_channels % groups != 0:
            raise ValueError("invalid in_channels size")
        if out_channels % groups != 0:
            raise ValueError("invalid in_channels size")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # define the bayesian parameters
        self.define_bayesian_parameters()

    def define_bayesian_parameters(self):
        """Define Bayesian Parameters."""
        self.mu_weight = Parameter(
            torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )
        self.rho_weight = Parameter(
            torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            )
        )
        self.register_buffer(
            "eps_weight",
            torch.randn(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            ),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_mu",
            torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            ),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size
            ),
            persistent=False,
        )

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(self.out_channels))
            self.rho_bias = Parameter(torch.Tensor(self.out_channels))
            self.register_buffer(
                "eps_bias", torch.randn(self.out_channels), persistent=False
            )
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(self.out_channels), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(self.out_channels), persistent=False
            )
        else:
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None)
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)

        super().init_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through conv.

        Args:
            x: input

        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout"
        """
        if self.is_frozen:
            eps_weight = self.eps_weight
        else:
            eps_weight = self.eps_weight.data.normal_()

        # compute variance of weight from unconstrained variable rho_kernel
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        # compute delta_weight
        delta_weight = sigma_weight * eps_weight

        bias = None
        delta_bias = None
        if self.bias:
            if self.is_frozen:
                eps_bias = self.eps_bias
            else:
                eps_bias = self.eps_bias.data.normal_()
            # compute variance of bias from unconstrained variable rho_bias
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            # compute delta_bias
            delta_bias = sigma_bias * eps_bias
            bias = self.mu_bias + delta_bias

        if self.layer_type == "reparameterization":
            weight = self.mu_weight + delta_weight
            out = self.conv_function(
                x, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            # linear outputs
            outputs = self.conv_function(
                x,
                weight=self.mu_weight,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            if self.is_frozen:
                torch.manual_seed(0)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # perturbed feedforward
            perturbed_outputs = (
                self.conv_function(
                    x * sign_input,
                    bias=delta_bias,
                    weight=delta_weight,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            # returning outputs + perturbations
            out = outputs + perturbed_outputs
        return out

    def extra_repr(self):
        """Representation when printing out layer."""
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, is_frozen={is_frozen}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)
