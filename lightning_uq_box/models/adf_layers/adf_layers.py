"""Utility functions for ADF Layers.

These are based on the library
https://github.com/uzh-rpg/deep_uncertainty_estimation/tree/master .
"""


from __future__ import absolute_import, division, print_function

import operator
from collections import OrderedDict
from itertools import islice

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from .adf_utils import normcdf, normpdf, resize2D_as


class AvgPool2d_adf(nn.Module):
    """Average Pool 2d Layer adapted for ADF."""

    def __init__(self, keep_variance_fn=None):
        """Initialize a new instance of a Average Pool 2d Layer adapted for ADF.

        Args:
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.AvgPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, inputs_mean, inputs_variance, kernel_size):
        outputs_mean = F.avg_pool2d(inputs_mean, kernel_size)
        outputs_variance = F.avg_pool2d(inputs_variance, kernel_size)
        outputs_variance = outputs_variance / (
            inputs_mean.size(2) * inputs_mean.size(3)
        )

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)

        # TODO: avg pooling means that every neuron is multiplied by the same
        #       weight, that is 1/number of neurons in the channel
        #      outputs_variance*1/(H*W) should be enough already

        return outputs_mean, outputs_variance


class Softmax_adf(nn.Module):
    """Softmax function adapted for ADF."""

    def __init__(self, dim: int = 1, keep_variance_fn=None):
        """Initialize a new instance of a Softmax function adapted for ADF.

        Args:
            dim: A dimension along which Softmax will be computed
            (so every slice along dim will sum to 1).
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.Softmax, self).__init__()
        self.dim = dim
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""

        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance * (
            torch.exp(features_variance) - 1
        )

        constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        constant = constant.unsqueeze(self.dim)
        outputs_mean = log_gaussian_mean / constant
        outputs_variance = log_gaussian_variance / (constant**2)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ReLU_adf(nn.Module):
    """ReLU function adapted for ADF."""

    def __init__(self, keep_variance_fn=None):
        """Initialize a new instance of a ReLU function adapted for ADF.

        Args:
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (
            (features_mean**2 + features_variance) * cdf
            + features_mean * features_stddev * pdf
            - outputs_mean**2
        )
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class LeakyReLU_adf(nn.Module):
    """Leaky ReLU function adapted for ADF."""

    def __init__(self, negative_slope: float = 0.01, keep_variance_fn=None):
        """Initialize a new instance of a Leaky ReLU function adapted for ADF.

        Args:
            negative_slope: Controls the angle of the negative slope
            (which is used for negative input values). Default: 1e-2
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean**2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r**2
        mean_n = -features_mean * negative_cdf + stddev_pdf
        variance_n = (
            squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n**2
        )
        covxy = -mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = (
            variance_r
            + self._negative_slope * self._negative_slope * variance_n
            - 2.0 * self._negative_slope * covxy
        )
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Dropout_adf(nn.Module):
    """Dropout for 2d layer adapted for ADF."""

    def __init__(self, p: float = 0.5, keep_variance_fn=None, inplace=False):
        """Initialize a new instance of a Dropout 2d layer adapted for ADF.

        Args:
            p: Dropout rate, probability of an element to be zeroed. Default: 0.5,
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.Dropout, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.inplace = inplace
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, inputs_mean, inputs_variance):
        if self.training:
            binary_mask = torch.ones_like(inputs_mean)
            binary_mask = F.dropout2d(binary_mask, self.p, self.training, self.inplace)

            outputs_mean = inputs_mean * binary_mask
            outputs_variance = inputs_variance * binary_mask**2

            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance

        outputs_variance = inputs_variance
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return inputs_mean, outputs_variance


class MaxPool2d_adf(nn.Module):
    """Max Pooling 2d layer adapted for ADF."""

    def __init__(self, keep_variance_fn=None):
        """Initialize a new instance of a Max Pooling 2d layer adapted for ADF.

        Args:
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = (
            (mu_a + mu_b) * stddev * pdf
            + (mu_a**2 + var_a) * cdf
            + (mu_b**2 + var_b) * (1.0 - cdf)
            - z_mu**2
        )
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b
        )
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b
        )
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance):
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return outputs_mean, outputs_variance


class Linear_adf(nn.Module):
    """Linear layer adapted for ADF."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        keep_variance_fn=None,
    ):
        """Initialize a new instance of a Linear layer adapted for ADF.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            bias: if set to False, the layer will not learn an additive bias,
            keep_variance_fn: propagation of input variance, function. Default: None,
        """

        super(nn.Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_variance, self.weight**2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class BatchNorm2d_adf(nn.Module):
    """Batch Norm 2d layer adapted for ADF."""

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        keep_variance_fn=None,
    ):
        """Initialize a new instance of a Batch Norm 2d layer adapted for ADF.

        Args:
            num_features: C from an expected input of size (N,C,H,W),
            eps: a value added to the denominator for numerical stability. Default: 1e-5,
            momentum: the value used for the running_mean and running_var computation.
            Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to True,
            this module has learnable affine parameters. Default: True,
            track_running_stats:  boolean value that when set to True, this module tracks the running mean
            and variance, and when set to False, this module does not track such statistics,
            and initializes statistics buffers running_mean and running_var as None.
            When these buffers are None, this module always uses batch statistics.
            in both training and eval modes. Default: True
            keep_variance_fn: propagation of input variance, function. Default: None,
        """
        super(nn.BatchNorm2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, inputs_mean, inputs_variance):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        outputs_mean = F.batch_norm(
            inputs_mean,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        outputs_variance = inputs_variance
        weight = ((self.weight.unsqueeze(0)).unsqueeze(2)).unsqueeze(3)
        outputs_variance = outputs_variance * weight**2
        """
        for i in range(outputs_variance.size(1)):
            outputs_variance[:,i,:,:]=outputs_variance[:,i,:,:].clone()*self.weight[i]**2
        """
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Conv2d_adf(_ConvNd):
    """Convolutional 2d layer adapted for ADF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        keep_variance_fn=None,
        padding_mode="zeros",
    ):
        """Initialize a new instance of a Convolutional 2d layer adapted for ADF.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            dilation: spacing between kernel elements. Default: 1,
            groups: controls the connections between inputs and outputs.
            in_channels and out_channels must both be divisible by groups,
            bias: if set to False, the layer will not learn an additive bias,
            keep_variance_fn: propagation of input variance, function. Default: None,
            padding_mode: Default: 'zeros',
        """
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(nn.Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.conv2d(
            inputs_mean,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        outputs_variance = F.conv2d(
            inputs_variance,
            self.weight**2,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ConvTranspose2d_adf(_ConvTransposeMixin, _ConvNd):
    """Convolutional Transpose 2d layer adapted for ADF."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        keep_variance_fn=None,
        padding_mode="zeros",
    ):
        """Initialize a new instance of a Convolutional Transpose 2d layer adapted for ADF.

        Args:
            in_channels: number of channels in the input image,
            out_channels: number of channels produced by the convolution,
            kernel_size: size of the convolving kernel,
            stride: stride of the convolution. Default: 1,
            padding: zero-padding added to both sides of the input. Default: 0,
            output_padding: Additional size added to one side of
            each dimension in the output shape. Default: 0,
            groups: controls the connections between inputs and outputs.
            in_channels and out_channels must both be divisible by groups,
            bias: if set to False, the layer will not learn an additive bias,
            dilation: spacing between kernel elements. Default: 1,
            keep_variance_fn: propagation of input variance, function. Default: None,
            padding_mode: Default: 'zeros',
        """
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(nn.ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, inputs_mean, inputs_variance, output_size=None):
        output_padding = self._output_padding(
            inputs_mean, output_size, self.stride, self.padding, self.kernel_size
        )
        outputs_mean = F.conv_transpose2d(
            inputs_mean,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        outputs_variance = F.conv_transpose2d(
            inputs_variance,
            self.weight**2,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    means = [resize2D_as(x[0], tensor_as[0], mode=mode) for x in tensor_list]
    variances = [resize2D_as(x[1], tensor_as[0], mode=mode) for x in tensor_list]
    means = torch.cat(means, dim=dim)
    variances = torch.cat(variances, dim=dim)
    return means, variances


class Sequential_adf(nn.Module):
    """Sequential container adapted for ADF."""

    def __init__(self, *args):
        """Initialize a new instance of a Sequential container adapted for ADF.

        Args:
            *args: OrderedDict[str, Module].
        """
        super(nn.Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return nn.Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(nn.Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, inputs, inputs_variance):
        for module in self._modules.values():
            inputs, inputs_variance = module(inputs, inputs_variance)

        return inputs, inputs_variance
