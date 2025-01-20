# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted for Lightning from https://github.com/tonyduan/mixture-density-network
# which is under MIT-License.

"""Mixture Density Layer for Regression."""

import torch
import torch.nn as nn
from torch import Tensor

from .mlp import MLP


class MixtureDensityLayer(nn.Module):
    """Mixture Density Network Layer."""

    valid_noise_types = ("diagonal", "isotropic", "isotropic_clusters", "fixed")

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_components: int,
        hidden_dims: list[int],
        noise_type: str = "diagonal",
        fixed_noise_level: None | float = None,
    ) -> None:
        """Initialize a new instance of Mixture Density Network Layer.

        Args:
            dim_in: dimensionality of the covariates
            dim_out: dimensionality of the response variable
            n_components: number of components in the mixture model
            hidden_dims: hidden dimension of the MDN layer
            noise_type: type of noise to model, choose one of
                ('diagonal', 'isotropic', 'isotropic_clusters', 'fixed')
            fixed_noise_level: in case of 'fixed' noise_type, specify the fixed noise
                level you want to use
        """
        assert noise_type in self.valid_noise_types, (
            f"Please choose one of {self.valid_noise_types}, you specified {noise_type}."
        )

        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type == "fixed")
        num_sigma_channels = {
            "diagonal": dim_out * n_components,
            "isotropic": n_components,
            "isotropic_clusters": 1,
            "fixed": 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level

        self.pi_network = MLP(
            n_inputs=dim_in, n_hidden=hidden_dims, n_outputs=n_components
        )
        self.normal_network = MLP(
            n_inputs=dim_in,
            n_hidden=hidden_dims,
            n_outputs=n_components * dim_out + num_sigma_channels,
        )

    def forward(self, x: Tensor, eps=1e-6) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of MDN network.

        Args:
            x: input tensor to MDN layer of dimension
                [batch_size, dim_in]
            eps: epsilon value for numerical stability

        Returns:
            log_pi [batch_size, n_components], mu [batch_size, n_components, dim_out], and sigma [batch_size, n_components, dim_out]
        """
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., : self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components :]
        if self.noise_type == "diagonal":
            sigma = torch.exp(sigma + eps)
        if self.noise_type == "isotropic":
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type == "isotropic_clusters":
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type == "fixed":
            assert self.fixed_noise_level is not None
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma
