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

"""Convolutional Variational Layers.

These are based on the Bayesian-torch library
https://github.com/IntelLabs/bayesian-torch (BSD-3 clause) but
adjusted to be trained with the Energy Loss.
"""

import torch.nn.functional as F

from .base_variational import BaseConvLayer_, BaseTransposeConvLayer_


class Conv1dVariational(BaseConvLayer_):
    """Convolutional 1D Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of Conv1dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 1
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            prior_mu,
            prior_sigma,
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
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of Conv2dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            prior_mu,
            prior_sigma,
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
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of Conv3dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 3
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bias,
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv3d


# Convoluational Transpose Layers


class ConvTranspose1dVariational(BaseTransposeConvLayer_):
    """Convolutional 1D Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        output_padding: int = 0,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of ConvTranspose1dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
            output_padding: additional size added to one side of the output shape
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 1
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bias,
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose1d


class ConvTranspose2dVariational(BaseTransposeConvLayer_):
    """Convolutional 2D Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        output_padding: int = 0,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of ConvTranspose2dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
            output_padding: additional size added to one side of the output shape
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bias,
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose2d


class ConvTranspose3dVariational(BaseTransposeConvLayer_):
    """Convolutional 3D Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        output_padding: int = 0,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
        layer_type: str = "reparameterization",
    ):
        """Initialize a new instance of ConvTranspose3dVariational layer.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: number of blocked connections
                from input channels to output channels,
            output_padding: additional size added to one side of the output shape
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
        """
        assert isinstance(kernel_size, tuple)
        assert len(kernel_size) == 3
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bias,
            layer_type,
        )
        # define what convolution to apply
        self.conv_function = F.conv_transpose3d
