"""Convolutional Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to reduce code duplication and to be trained with the Energy Loss.
"""

import torch.nn.functional as F

from .base_variational import BaseConvLayer_


class Conv1dVariational(BaseConvLayer_):
    """Convolutional 1D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of Conv1dVariational layer.

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
            layer_type,
        )

        # define what convolution to apply
        self.conv_function = F.conv1d


class Conv2dVariational(BaseConvLayer_):
    """Convolutional 2D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of Conv2dVariational layer.

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
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv2d


class Conv3dVariational(BaseConvLayer_):
    """Convolutional 2D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of Conv3dVariational layer.

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
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv3d


# Convoluational Transpose Layers


class ConvTranspose1dVariational(BaseConvLayer_):
    """Convolutional 2D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of ConvTranspose1dVariational layer.

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
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose1d


class ConvTranspose2dVariational(BaseConvLayer_):
    """Convolutional 2D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of ConvTranspose2dVariational layer.

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
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose2d


class ConvTranspose3dVariational(BaseConvLayer_):
    """Convolutional 3D Variational Layer adapted for Alpha Divergence."""

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
        """Initialize a new instance of ConvTranspose3dVariational layer.

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
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose3d
