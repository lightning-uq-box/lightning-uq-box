# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted from Reference Implementation: https://github.com/yookoon/density_uncertainty_layers

"""Density Uncertainty Layers."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .masked_conv import MaskedConv2d


class DensityLayerBase_(nn.Module):
    """Base class for Density Layers.

    This class provides common methods for sampling, computing log probability density function (logpdf),
    and computing Kullback-Leibler (KL) divergence. These methods can be used by derived classes to
    implement specific density layers.
    """

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Generate a sample from a Gaussian distribution with given mean and log variance.

        Args:
            mu: Mean of the Gaussian distribution.
            logvar: Log variance of the Gaussian distribution.

        Returns:
            A sample from the Gaussian distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def logpdf(self, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute the log probability density function of a Gaussian distribution.

        Args:
            x: Input tensor.
            mu: Mean of the Gaussian distribution.
            logvar: Log variance of the Gaussian distribution.

        Returns:
            The log probability density function value.
        """
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + logvar + ((x - mu) ** 2) / torch.exp(logvar)
        )

    def kl_div(
        self, mu1: Tensor, logvar1: Tensor, mu2: Tensor, logvar2: Tensor
    ) -> Tensor:
        """Compute the Kullback-Leibler (KL) divergence between two Gaussian distributions.

        Args:
            mu1: Mean of the first Gaussian distribution.
            logvar1: Log variance of the first Gaussian distribution.
            mu2: Mean of the second Gaussian distribution.
            logvar2: Log variance of the second Gaussian distribution.

        Returns:
            The KL divergence value.
        """
        return 0.5 * torch.sum(
            logvar2
            - logvar1
            - 1.0
            + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2)
        )

    def compute_kl_div(self):
        """Compute the KL divergence between the prior and the posterior."""
        s_kl = self.kl_div(
            self.s_prior_mu, self.s_prior_logvar, self.s_prior_mu, self.s_logvar
        )
        b_kl = self.kl_div(
            self.b_prior_mu, self.b_prior_logvar, self.b_prior_mu, self.b_logvar
        )
        return s_kl + b_kl


class DensityLinear(DensityLayerBase_):
    """Linear Density Uncertainty Layer.

    If you use this module in your work, please cite the following paper:

    * https://arxiv.org/abs/2306.12497
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_std: float = 0.1,
        posterior_std_init: float = 1e-3,
    ) -> None:
        """Initialize a Linear Density Layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: If True, add a bias term.
            prior_std: Standard deviation of the prior.
            posterior_std_init: Initial standard deviation of the posterior.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight)

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer("s_prior_mu", torch.zeros(1, out_features))
        self.register_buffer(
            "s_prior_logvar", prior_logvar * torch.ones(1, out_features)
        )
        self.register_buffer("b_prior_mu", torch.zeros(1, out_features))
        self.register_buffer(
            "b_prior_logvar", prior_logvar * torch.ones(1, out_features)
        )

        self.s_logvar = nn.Parameter(
            2.0 * np.log(posterior_std_init) * torch.ones(1, out_features)
        )
        self.b_logvar = nn.Parameter(
            2.0 * np.log(posterior_std_init) * torch.ones(1, out_features)
        )

        # Initialize activation covariance estimates
        self.L = nn.Parameter(torch.zeros(in_features, in_features))
        self.register_buffer("I", torch.eye(in_features))
        # Diagonal log variance
        self.logvar = nn.Parameter(torch.zeros(1, in_features))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the linear layer.

        Args:
            x: Input tensor of shape [batch_size, in_features].

        Returns:
            Output tensor of shape [batch_size, out_features].
        """
        D = x.shape[1]

        L = self.L.tril(diagonal=-1) + self.I
        z = x.detach() @ L
        Ex = torch.sum(z**2 / self.logvar.exp(), 1, keepdim=True) / 2
        self.loglikelihood = -0.5 * (
            D * np.log(2 * np.pi) + self.logvar.sum()
        ) - Ex.mean(dim=1)

        # Energy can fluctuate wildly during training so apply clipping
        Ex = Ex.clip(0, D)
        # Energy will be D/2 on average. Scale the noise bias term to match their scales
        noise_var = self.s_logvar.exp() * Ex.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)

        a = self.linear(x)
        a = a + noise_std * torch.rand_like(a)
        return a


class DensityConv2d(DensityLayerBase_):
    """Conv2d Density Uncertainty Layer.

    If you use this module in your work, please cite the following paper:

    * https://arxiv.org/abs/2306.12497
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int,
        padding: int = 0,
        bias: bool = True,
        prior_std: float = 0.1,
        posterior_std_init: float = 1e-3,
    ):
        """Initialize a Conv2d Density Layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            bias: If True, add a bias term.
            prior_std: Standard deviation of the prior.
            posterior_std_init: Initial standard deviation of the posterior.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        nn.init.kaiming_normal_(self.conv.weight)

        prior_logvar = 2.0 * math.log(prior_std)
        self.register_buffer("s_prior_mu", torch.zeros(1, out_channels, 1, 1))
        self.register_buffer(
            "s_prior_logvar", prior_logvar * torch.ones(1, out_channels, 1, 1)
        )
        self.register_buffer("b_prior_mu", torch.zeros(1, out_channels, 1, 1))
        self.register_buffer("b_prior_logvar", torch.ones(1, out_channels, 1, 1))

        self.s_logvar = nn.Parameter(
            2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1)
        )
        self.b_logvar = nn.Parameter(
            2.0 * np.log(posterior_std_init) * torch.ones(1, out_channels, 1, 1)
        )

        # Generative
        self.masked_conv = MaskedConv2d(in_channels, kernel_size, padding=padding)
        self.logvar = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the convolutional layer.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Output tensor of shape [batch_size, out_channels, height, width].
        """
        # x: [B, D, H, W]
        D = x.shape[1]

        z = x.detach() + self.masked_conv(x.detach())
        Ex = torch.sum(z**2 / self.logvar.exp(), dim=1, keepdim=True) / 2
        self.loglikelihood = -0.5 * (
            D * np.log(2 * np.pi) + self.logvar.sum()
        ) - Ex.mean(dim=[1, 2, 3])
        # Average pool the energy in the local convolutional window
        Ex_pool = self.pool(Ex)

        a = self.conv(x)

        Ex_pool = Ex_pool.clip(0, D)
        noise_var = self.s_logvar.exp() * Ex_pool.detach() + self.b_logvar.exp() * D / 2
        noise_std = torch.sqrt(noise_var + 1e-16)

        a = a + noise_std * torch.rand_like(a)
        return a
