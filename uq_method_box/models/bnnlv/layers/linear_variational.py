"""Linear Variational Layer adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers.linear_variational import (
    LinearReparameterization,
)
from torch import Tensor

from .utils import calc_log_f_hat, calc_log_normalizer


class LinearVariational(LinearReparameterization):
    """Linear Variational Layer adapted for Alpha Divergence."""

    valid_layer_types = ["reparameterization", "flipout"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean=0,
        prior_variance=1,
        posterior_mu_init=0,
        posterior_rho_init=-3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers.linear_variational,
        LinearReparameterization. Works for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_features: size of each input sample,
            out_features: size of each output sample,
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
                Default: True.
            layer_type: reparameterization trick with
                "reparameterization" or "flipout".
        """
        super().__init__(
            in_features,
            out_features,
            prior_mean,
            prior_variance,
            posterior_mu_init,
            posterior_rho_init,
            bias,
        )
        assert (
            layer_type in self.valid_layer_types
        ), f"Only {self.valid_layer_types} are valid layer types but found {layer_type}"
        self.layer_type = layer_type

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_weight.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Returns:
            log_normalizer.
        """
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
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = sigma_weight * self.eps_weight.data.normal_()

        # sampling weight and bias
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
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + calc_log_f_hat(
                w=bias,
                m_W=self.mu_bias,
                std_W=sigma_bias,
                prior_variance=self.prior_variance,
            )

        return log_f_hat

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through layer.

        Args:
            x: input.

        Returns:
            outputs of variational layer
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        # compute bias if available
        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())

        # forward pass with chosen layer type
        if self.layer_type == "reparameterization":
            weight = self.mu_weight + (sigma_weight * self.eps_weight.data.normal_())
            output = F.linear(x, weight, bias)
        else:
            # sampling delta_W and delta_b
            sigma_weight = torch.log1p(torch.exp(self.rho_weight))
            delta_weight = sigma_weight * self.eps_weight.data.normal_()
            # linear outputs
            out = F.linear(x, self.mu_weight, self.mu_bias)
            # flipout
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = out.clone().uniform_(-1, 1).sign()
            output = out + (F.linear(x * sign_input, delta_weight, bias) * sign_output)

        return output
