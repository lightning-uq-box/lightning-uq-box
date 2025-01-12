# BSD 3-Clause License

# Copyright (c) 2024, Intel Labs
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

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Linear Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to be trained with the Energy Loss and have batched samples.
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
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
        batched_samples: bool = False,
        max_n_samples: int | None = None,
    ):
        """Initialize a new instance of LinearVariational layer.

        Args:
            in_features: size of each input sample,
            out_features: size of each output sample,
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
            batched_samples: bool to allow batched sampling which can
                provide significant speed ups iff your BNN is a model
                with just nn.Linear layers
            max_n_samples: maximum number of samples you intend to draw
                with a single forward pass, needs to be specified with
                *batched_samples* is True
        """
        super().__init__(
            prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init, bias
        )

        if batched_samples:
            assert isinstance(max_n_samples, int), (
                "If you use `batched_samples`, you need to specify `max_n_samples`."
            )
            self.max_n_samples = max_n_samples

        assert layer_type in self.valid_layer_types, (
            f"Only {self.valid_layer_types} are valid layer types but found {layer_type}"
        )
        self.layer_type = layer_type

        self.in_features = in_features
        self.out_features = out_features
        self.batched_samples = batched_samples
        if not self.batched_samples:
            self.max_n_samples = 1

        # creat and initialize bayesian parameters
        self.define_bayesian_weight_params()

    def define_bayesian_weight_params(self) -> None:
        """Define Bayesian parameters."""
        self.mu_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.rho_weight = Parameter(torch.Tensor(self.out_features, self.in_features))

        self.register_buffer(
            "eps_weight",
            torch.randn(self.max_n_samples, self.out_features, self.in_features),
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
                "eps_bias",
                torch.randn(self.max_n_samples, self.out_features),
                persistent=False,
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
        if self.batched_samples:
            n_samples = x.shape[0]
            assert n_samples <= self.max_n_samples, (
                "Number of samples needs to be <= max_n_samples"
                f" but found max_n_samples {self.max_n_samples} "
                f" and {n_samples} in input tensor."
            )
            assert x.dim() == 3, (
                "Expect input to be a tensor of shape "
                "[num_samples, batch_size, num_features], "
                f"but found shape {x.shape}"
            )
        else:
            n_samples = 1

        delta_weight, delta_bias = self.sample_weights(n_samples)

        # forward pass with chosen layer type
        if self.layer_type == "reparameterization":
            # sample weight via reparameterization trick
            output = x.matmul((self.mu_weight + delta_weight).transpose(-1, -2))
            if self.bias:
                output = output + (self.mu_bias + delta_bias).unsqueeze(1)
        else:
            # linear outputs
            out = F.linear(x, self.mu_weight, self.mu_bias)
            # flipout
            if self.is_frozen:
                torch.manual_seed(0)
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = out.clone().uniform_(-1, 1).sign()
            # get outputs+perturbed outputs
            output = out + (
                (
                    (x * sign_input).matmul(delta_weight.transpose(-1, -2))
                    + delta_bias.unsqueeze(1)
                )
                * sign_output
            )
        if not self.batched_samples:
            output = output.squeeze(0)
        return output

    def sample_weights(self, n_samples: int) -> tuple[Tensor]:
        """Sample variational weights for batched sampling.

        Args:
            n_samples: number of samples to draw

        Returns:
            delta_weight and delta_bias
        """
        if self.is_frozen:
            eps_weight = self.eps_weight
            bias_eps = self.eps_bias
        else:
            eps_weight = self.eps_weight.data.normal_()
            if self.mu_bias is not None:
                bias_eps = self.eps_bias.data.normal_()

        # select from max_samples
        eps_weight = eps_weight[:n_samples]

        # select first sample if not batched to keep consistent shape

        # sample weight with reparameterization trick
        sigma_weight = F.softplus(self.rho_weight)
        delta_weight = eps_weight * sigma_weight

        delta_bias = torch.zeros(1).to(self.rho_weight.device)

        if self.mu_bias is not None:
            bias_eps = bias_eps[:n_samples]

            # sample bias with reparameterization trick
            sigma_bias = F.softplus(self.rho_bias)
            delta_bias = bias_eps * sigma_bias
        return delta_weight, delta_bias

    def freeze_layer(self, n_samples: int | None = None) -> None:
        """Freeze Variational Layers.

        This is useful when using BNN+LV to fix the BNN parameters
        to sample the Latent Variables to estimate aleatoric uncertainy.

        Args:
            n_samples: number of samples to fix for batched approach
        """
        self.is_frozen = True
        if n_samples is not None:
            if self.max_n_samples < n_samples:
                self.max_n_samples = n_samples

        setattr(
            self,
            "eps_weight",
            torch.randn(self.max_n_samples, self.out_features, self.in_features),
        )
        setattr(self, "eps_bias", torch.randn(self.max_n_samples, self.out_features))

    def extra_repr(self) -> str:
        """Representation when printing out Layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, is_frozen={self.is_frozen}"
