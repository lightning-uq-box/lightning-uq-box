"""Convolutional Variational Layers adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers.conv_variational import (
    Conv1dReparameterization,
    Conv2dReparameterization,
    Conv3dReparameterization,
    ConvTranspose1dReparameterization,
    ConvTranspose2dReparameterization,
    ConvTranspose3dReparameterization,
)
from torch import Tensor

from .utils import calc_log_f_hat, calc_log_normalizer


class Conv1dVariational(Conv1dReparameterization):
    """Convolutional Variational Layer adapted for Alpha Divergence."""

    valid_layer_types = ["reparameterization", "flipout"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement Conv1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers,
        Conv1dReparameterization. Works for for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for kernel weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv1d(
                x, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

            return out

        if self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv1d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv1d(
                    x * sign_input,
                    bias=bias,
                    weight=delta_kernel,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            # returning outputs + perturbations
            return outputs + perturbed_outputs


class Conv2dVariational(Conv2dReparameterization):
    """Convolutional Variational Layers adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement Conv2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers,
        Conv2dReparameterization. Works for for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input.
                Default: 0,
            dilation: spacing between kernel elements.
                Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
            prior_mean: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_variance: variance of the prior
                arbitrary distribution to be used on the complexity cost,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv2d(
                x, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

            return out

        if self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv2d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv2d(
                    x * sign_input,
                    bias=bias,
                    weight=delta_kernel,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            # returning outputs + perturbations
            return outputs + perturbed_outputs


class Conv3dVariational(Conv3dReparameterization):
    """Convolutional Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement Conv3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers,
        Conv3dReparameterization. Works for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1.
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv3d(
                x, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

            return out

        if self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv3d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv3d(
                    x * sign_input,
                    bias=bias,
                    weight=delta_kernel,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            # returning outputs + perturbations
            return outputs + perturbed_outputs


class ConvTranspose1dVariational(ConvTranspose1dReparameterization):
    """Convolutional Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: type = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement ConvTranspose1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers,
        ConvTranspose1dReparameterization. Works for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input.
                Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv_transpose1d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            )

            return out

        elif self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv_transpose1d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv_transpose1d(
                    x * sign_input,
                    weight=delta_kernel,
                    bias=bias,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            return outputs + perturbed_outputs


class ConvTranspose2dVariational(ConvTranspose2dReparameterization):
    """Convolutional Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement ConvTranspose2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers,
        ConvTranspose2dReparameterization. Works for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements.
                Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
            type: reparameterization trick with
                "reparameterization" or "flipout".
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv_transpose2d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            )

            return out

        elif self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv_transpose2d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv_transpose2d(
                    x * sign_input,
                    weight=delta_kernel,
                    bias=bias,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            return outputs + perturbed_outputs


class ConvTranspose3dVariational(ConvTranspose3dReparameterization):
    """Convolutional Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """
        Implement ConvTranspose3d layer.

        With reparameterization trick.
        Inherits from bayesian_torch.layers.variational_layers,
        ConvTranspose3dReparameterization. Works for Reparameterization
        or Flipout reparameterization.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution.
                Default: 1,
            padding: zero-padding added to both sides of the input.
                Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
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
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = calc_log_normalizer(m_W=self.mu_kernel, std_W=sigma_weight)

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + calc_log_normalizer(
                m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_normalizer

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        delta_weight = sigma_weight * self.eps_kernel.data.normal_()

        # sampling weight and bias
        weight = self.mu_kernel + delta_weight

        # get log_f_hat for weights
        log_f_hat = calc_log_f_hat(
            w=weight,
            m_W=self.mu_kernel,
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

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer if type="reparameterization"
            outputs+perturbed of layer for type="flipout".
        """
        if self.layer_type == "reparameterization":
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()
            weight = self.mu_kernel + (sigma_weight * eps_kernel)

            bias = None

            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = self.mu_bias + (sigma_bias * eps_bias)

            out = F.conv_transpose3d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            )

            return out

        elif self.layer_type == "flipout":
            # linear outputs
            outputs = F.conv_transpose3d(
                x,
                weight=self.mu_kernel,
                bias=self.mu_bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            # sampling perturbation signs
            sign_input = x.clone().uniform_(-1, 1).sign()
            sign_output = outputs.clone().uniform_(-1, 1).sign()

            # gettin perturbation weights
            sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
            eps_kernel = self.eps_kernel.data.normal_()

            delta_kernel = sigma_weight * eps_kernel

            bias = None
            if self.bias:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                eps_bias = self.eps_bias.data.normal_()
                bias = sigma_bias * eps_bias

            # perturbed feedforward
            perturbed_outputs = (
                F.conv_transpose3d(
                    x * sign_input,
                    weight=delta_kernel,
                    bias=bias,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                * sign_output
            )

            return outputs + perturbed_outputs
