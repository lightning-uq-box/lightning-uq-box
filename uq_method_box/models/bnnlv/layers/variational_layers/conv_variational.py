"""Convolutional Variational Layers adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers.conv_variational import *
from torch import Tensor

__all__ = [
    "Conv1dReparameterization",
    "Conv2dReparameterization",
    "Conv3dReparameterization",
    "ConvTranspose1dReparameterization",
    "ConvTranspose2dReparameterization",
    "ConvTranspose3dReparameterization",
]


class Conv1dReparameterization(Conv1dReparameterization):
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
        bias=True,
    ):
        """
        Implements Conv1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers Conv1dReparameterization

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for kernel weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed outputs of layer, log_f_hat, log_normalizer.
        """
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

        return out


class Conv2dReparameterization(Conv2dReparameterization):
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
        bias=True,
    ):
        """
        Implements Conv2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers, Conv2dReparameterization.

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

        return out


class Conv3dReparameterization(Conv3dReparameterization):
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
        bias=True,
    ):
        """
        Implements Conv3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.flipout_layers, Conv3dReparameterization.

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

        return out


class ConvTranspose1dReparameterization(ConvTranspose1dReparameterization):
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
        bias=True,
    ):
        """
        Implements ConvTranspose1d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers, ConvTranspose1dReparameterization.

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv_transpose1d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

        return out


class ConvTranspose2dReparameterization(ConvTranspose2dReparameterization):
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
        bias=True,
    ):
        """
        Implements ConvTranspose2d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers, ConvTranspose2dReparameterization.

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv_transpose2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

        return out


class ConvTranspose3dReparameterization(ConvTranspose3dReparameterization):
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
        bias=True,
    ):
        """
        Implements ConvTranspose3d layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers, ConvTranspose3dReparameterization.

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
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

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_f_hat, log_normalizer.
        """

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_kernel, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
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
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_kernel, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)

        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)

        out = F.conv_transpose3d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

        return out
