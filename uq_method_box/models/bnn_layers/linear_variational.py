"""Linear Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to reduce code duplication and to be trained with the Energy Loss.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from .base_variational import BaseVariationalLayer_


class LinearVariational(BaseVariationalLayer_):
    """Linear Variational Layer adapted for Alpha Divergence."""

    valid_layer_types = ["reparameterization", "flipout"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of LinearVariational layer.

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
            layer_type: reparameterization trick with
                "reparameterization" or "flipout".
        """
        super().__init__(
            prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias
        )

        assert (
            layer_type in self.valid_layer_types
        ), f"Only {self.valid_layer_types} are valid layer types but found {layer_type}"
        self.layer_type = layer_type

        self.in_features = in_features
        self.out_features = out_features

        # creat and initialize bayesian parameters
        self.define_bayesian_parameters()

    def define_bayesian_parameters(self) -> None:
        """Define Bayesian parameters."""
        self.mu_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.rho_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_buffer(
            "eps_weight",
            torch.Tensor(self.out_features, self.in_features),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_mu",
            torch.Tensor(self.out_features, self.in_features),
            persistent=False,
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(self.out_features, self.in_features),
            persistent=False,
        )
        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(self.out_features))
            self.rho_bias = Parameter(torch.Tensor(self.out_features))
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(self.out_features), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(self.out_features), persistent=False
            )
            self.register_buffer(
                "eps_bias", torch.Tensor(self.out_features), persistent=False
            )
        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None, persistent=False)

        super().init_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through layer.

        Args:
            x: input.

        Returns:
            outputs of variational layer
        """
        # compute variance of weight from unconstrained variable rho_weight
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        # compute bias and delta_bias if available
        bias = None
        if self.freeze:
            eps_weight = self.eps_weight
        else:
            eps_weight = self.eps_weight.data.normal_()


        if self.mu_bias is not None:
            if self.freeze:
                eps_bias = self.eps_bias
            else:
                eps_bias = self.eps_bias.data.normal_()


            # compute variance of bias from unconstrained variable rho_bias
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * eps_bias
            bias = self.mu_bias + delta_bias

        # forward pass with chosen layer type
        if self.layer_type == "reparameterization":
            # sample weight via reparameterization trick
            weight = self.mu_weight + (sigma_weight * eps_weight)
            output = F.linear(x, weight, bias)
        else:
            # sampling delta_W
            delta_weight = sigma_weight * eps_weight
            # linear outputs
            out = F.linear(x, self.mu_weight, self.mu_bias)
            # flipout
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = out.clone().uniform_(-1, 1).sign()
            # get outputs+perturbed outputs
            output = out + (
                F.linear(x * sign_input, delta_weight, delta_bias) * sign_output
            )

        return output
