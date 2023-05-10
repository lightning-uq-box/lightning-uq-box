"""Cpnvolutional Flipout Layers adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.conv_flipout import *
from torch import Tensor

__all__ = [
    "Conv1dFlipout",
    "Conv2dFlipout",
    "Conv3dFlipout",
    "ConvTranspose1dFlipout",
    "ConvTranspose2dFlipout",
    "ConvTranspose3dFlipout",
]


class Conv1dFlipout(Conv1dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""

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
    ):
        """
        Implement Conv1d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers Conv1dFlipout

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
            prior_mean: mean of the prior arbitrary distribution
                to be used on the complexity cost,
            prior_variance: variance of the prior arbitrary
                distribution to be used on the complexity cost,
            posterior_mu_init: init trainable mu parameter
                representing mean of the approximate posterior,
            posterior_rho_init: init trainable rho parameter
                representing the sigma of the approximate posterior
                through softplus function,
            bias: if set to False, the layer will not
                learn an additive bias. Default: True,
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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for kernel weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
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


class Conv2dFlipout(Conv2dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""
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
    ):
        """
        Implement Conv2d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers, Conv2dFlipout.

        Parameters:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input.
                Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections from
                input channels to output channels,
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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
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


class Conv3dFlipout(Conv3dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""

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
    ):
        """
        Implement Conv3d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers, Conv3dFlipout.

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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat


def forward(self, x):
    """Forward pass through layer.

    Args: self: layer.
        x: input.
    Returns:
        outputs+perturbed of layer.
    """
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


class ConvTranspose1dFlipout(ConvTranspose1dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""

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
    ):
        """
        Implement ConvTranspose1d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers, ConvTranspose1dFlipout.

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
            prior_variance: variance of the prior arbitrary
                distribution to be used on the complexity cost,
            posterior_mu_init: init trainable mu parameter
                representing mean of the approximate posterior,
            posterior_rho_init: init trainable rho parameter
                representing the sigma of the approximate
                    posterior through softplus function,
            bias: if set to False, the layer will not learn an additive bias.
                Default: True.
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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
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


class ConvTranspose2dFlipout(ConvTranspose2dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""

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
    ):
        """
        Implement ConvTranspose2d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers,
        ConvTranspose2dFlipout.

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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
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


class ConvTranspose3dFlipout(ConvTranspose3dFlipout):
    """Cpnvolutional Flipout Layer adapted for Alpha Divergence."""

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
    ):
        """
        Implement ConvTranspose3d layer.

        With Flipout reparameterization trick.
        Inherits from bayesian_torch.layers.variational_layers,
        ConvTranspose3dFlipout.

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
            prior_variance: variance of the prior
                arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: init trainable mu parameter
                representing mean of the approximate posterior,
            posterior_rho_init: init trainable rho
                parameter representing the sigma of the
                    approximate posterior through softplus function,
            bias: if set to False, the layer will not learn an additive bias.
                Default: True.
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

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_kernel.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        log_normalizer = self.calc_log_normalizer(
            m_W=self.mu_kernel, std_W=sigma_weight
        )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            log_normalizer = log_normalizer + self.calc_log_normalizer(
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
        log_f_hat = self.calc_log_f_hat(
            w=weight, m_W=self.mu_kernel, std_W=sigma_weight
        )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            log_f_hat = log_f_hat + self.calc_log_f_hat(
                w=bias, m_W=self.mu_bias, std_W=sigma_bias
            )

        return log_f_hat

    def forward(self, x):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
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
