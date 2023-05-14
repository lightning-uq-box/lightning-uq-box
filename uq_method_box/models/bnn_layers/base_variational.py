"""Base Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to reduce code duplication and to be trained with the Energy Loss.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from .utils import calc_log_f_hat, calc_log_normalizer


class BaseVariationalLayer_(nn.Module):
    """Base Variational Layer for BNN Layers."""

    valid_layer_types = ["reparameterization", "flipout"]

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ) -> None:
        """Initialize a new instance of Base Variational Layer.

        Args:
            prior_mean: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_variance: variance of the prior arbitrary
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

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        assert (
            layer_type in self.valid_layer_types
        ), f"Only {self.valid_layer_types} are valid layer types but found {layer_type}"
        self.layer_type = layer_type
        self.freeze = False


    def define_bayesian_parameters(self):
        """Define Bayesian parameters."""
        raise NotImplementedError

    def init_parameters(self):
        """Initialize Bayesian Parameters."""
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)
        if self.bias:
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_weight.numel() + self.mu_weight.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
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
            prior_variance=self.prior_variance,
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
                w=bias,
                m_W=self.mu_bias,
                std_W=sigma_bias,
                prior_variance=self.prior_variance,
            )

        return log_f_hat


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
        prior_mean: float = 0,
        prior_variance: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -3,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ) -> None:
        """Initialize a new instance of BaseConvLayer.

        Args:
            prior_mean: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_variance: variance of the prior arbitrary
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
            prior_mean,
            prior_variance,
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
            torch.Tensor(
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
                "eps_bias", torch.Tensor(self.out_channels), persistent=False
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

                
        if self.freeze:
            eps_weight = self.eps_weight
        else:
            eps_weight = self.eps_weight.data.normal_()
        
        # compute variance of weight from unconstrained variable rho_kernel
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        # compute delta_weight
        delta_weight = sigma_weight * eps_weight

        bias = None



        if self.bias:
            if self.freeze:
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
