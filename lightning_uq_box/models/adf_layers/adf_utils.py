from __future__ import absolute_import, division, print_function

import math
from numbers import Number
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as tf
from torch import Tensor, nn

import lightning_uq_box.models.adf_layers as adf_layers


# Tested against Matlab: Works correctly!
def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))


def _normal_log_pdf(value, mu, stddev):
    var = stddev**2
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(stddev)
    return -((value - mu) ** 2) / (2.0 * var) - log_scale - np.log(np.sqrt(2.0 * np.pi))


# Tested against Matlab: Works correctly!
def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))


def _bchw2bhwc(tensor):
    return tensor.transpose(1, 2).transpose(2, 3)


def _bhwc2bchw(tensor):
    return tensor.transpose(2, 3).transpose(1, 2)


class Meshgrid(nn.Module):
    def __init__(self):
        super(Meshgrid, self).__init__()
        self.width = 0
        self.height = 0
        self.register_buffer("xx", torch.zeros(1, 1))
        self.register_buffer("yy", torch.zeros(1, 1))
        self.register_buffer("rangex", torch.zeros(1, 1))
        self.register_buffer("rangey", torch.zeros(1, 1))

    def _compute_meshgrid(self, width, height):
        torch.arange(0, width, out=self.rangex)
        torch.arange(0, height, out=self.rangey)
        self.xx = self.rangex.repeat(height, 1).contiguous()
        self.yy = self.rangey.repeat(width, 1).t().contiguous()

    def forward(self, width, height):
        if self.width != width or self.height != height:
            self._compute_meshgrid(width=width, height=height)
            self.width = width
            self.height = height
        return self.xx, self.yy


class BatchSub2Ind(nn.Module):
    def __init__(self):
        super(BatchSub2Ind, self).__init__()
        self.register_buffer("_offsets", torch.LongTensor())

    def forward(self, shape, row_sub, col_sub, out=None):
        batch_size = row_sub.size(0)
        height, width = shape
        ind = row_sub * width + col_sub
        torch.arange(batch_size, out=self._offsets)
        self._offsets *= height * width

        if out is None:
            return torch.add(ind, self._offsets.view(-1, 1, 1))
        else:
            torch.add(ind, self._offsets.view(-1, 1, 1), out=out)


class Interp2(nn.Module):
    def __init__(self, clamp=False):
        super(Interp2, self).__init__()
        self._clamp = clamp
        self._batch_sub2ind = BatchSub2Ind()
        self.register_buffer("_x0", torch.LongTensor())
        self.register_buffer("_x1", torch.LongTensor())
        self.register_buffer("_y0", torch.LongTensor())
        self.register_buffer("_y1", torch.LongTensor())
        self.register_buffer("_i00", torch.LongTensor())
        self.register_buffer("_i01", torch.LongTensor())
        self.register_buffer("_i10", torch.LongTensor())
        self.register_buffer("_i11", torch.LongTensor())
        self.register_buffer("_v00", torch.FloatTensor())
        self.register_buffer("_v01", torch.FloatTensor())
        self.register_buffer("_v10", torch.FloatTensor())
        self.register_buffer("_v11", torch.FloatTensor())
        self.register_buffer("_x", torch.FloatTensor())
        self.register_buffer("_y", torch.FloatTensor())

    def forward(self, v, xq, yq):
        batch_size, channels, height, width = v.size()

        # clamp if wanted
        if self._clamp:
            xq.clamp_(0, width - 1)
            yq.clamp_(0, height - 1)

        # ------------------------------------------------------------------
        # Find neighbors
        #
        # x0 = torch.floor(xq).long(),          x0.clamp_(0, width - 1)
        # x1 = x0 + 1,                          x1.clamp_(0, width - 1)
        # y0 = torch.floor(yq).long(),          y0.clamp_(0, height - 1)
        # y1 = y0 + 1,                          y1.clamp_(0, height - 1)
        #
        # ------------------------------------------------------------------
        torch.clamp(torch.floor(xq).long(), 0, width - 1, out=self._x0)
        torch.clamp(torch.floor(yq).long(), 0, height - 1, out=self._y0)

        torch.clamp(torch.add(self._x0, 1), 0, width - 1, out=self._x1)
        torch.clamp(torch.add(self._y0, 1), 0, height - 1, out=self._y1)

        # batch_sub2ind
        self._batch_sub2ind([height, width], self._y0, self._x0, out=self._i00)
        self._batch_sub2ind([height, width], self._y0, self._x1, out=self._i01)
        self._batch_sub2ind([height, width], self._y1, self._x0, out=self._i10)
        self._batch_sub2ind([height, width], self._y1, self._x1, out=self._i11)

        # reshape
        v_flat = _bchw2bhwc(v).contiguous().view(-1, channels)
        torch.index_select(v_flat, dim=0, index=self._i00.view(-1), out=self._v00)
        torch.index_select(v_flat, dim=0, index=self._i01.view(-1), out=self._v01)
        torch.index_select(v_flat, dim=0, index=self._i10.view(-1), out=self._v10)
        torch.index_select(v_flat, dim=0, index=self._i11.view(-1), out=self._v11)

        # local_coords
        torch.add(xq, -self._x0.float(), out=self._x)
        torch.add(yq, -self._y0.float(), out=self._y)

        # weights
        w00 = torch.unsqueeze((1.0 - self._y) * (1.0 - self._x), dim=1)
        w01 = torch.unsqueeze((1.0 - self._y) * self._x, dim=1)
        w10 = torch.unsqueeze(self._y * (1.0 - self._x), dim=1)
        w11 = torch.unsqueeze(self._y * self._x, dim=1)

        def _reshape(u):
            return _bhwc2bchw(u.view(batch_size, height, width, channels))

        # values
        values = (
            _reshape(self._v00) * w00
            + _reshape(self._v01) * w01
            + _reshape(self._v10) * w10
            + _reshape(self._v11) * w11
        )

        if self._clamp:
            return values
        else:
            #  find_invalid
            invalid = (
                ((xq < 0) | (xq >= width) | (yq < 0) | (yq >= height))
                .unsqueeze(dim=1)
                .float()
            )
            # maskout invalid
            transformed = invalid * torch.zeros_like(values) + (1.0 - invalid) * values

        return transformed


def resize2D(inputs, size_targets, mode="bilinear"):
    size_inputs = [inputs.size(2), inputs.size(3)]

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = tf.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = tf.upsample(inputs, size=size_targets, mode=mode)  # upsampling

    # correct scaling
    return resized


def resize2D_as(inputs, output_as, mode="bilinear"):
    size_targets = [output_as.size(2), output_as.size(3)]
    return resize2D(inputs, size_targets, mode=mode)


class InverseWarpImage(nn.Module):
    def __init__(self, clamp=True):
        super(InverseWarpImage, self).__init__()
        self._meshgrid = Meshgrid()
        self._interp2 = Interp2(clamp=clamp)
        self._clamp = clamp

    def forward(self, image, u, v):
        height, width = image.size()[2:]
        xx, yy = self._meshgrid(width=width, height=height)
        xq = xx + torch.squeeze(u, dim=1)
        yq = yy + torch.squeeze(v, dim=1)

        return self._interp2(image, xq, yq)


def softmax_adf_layer(params: dict[str, Any], softmax_layer: nn.Softmax) -> nn.Module:
    """Convert Softmax deterministic layer to Softmax adf layer."""

    layer = softmax_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    softmax_adf_layer = layer_fn(**params)
    return softmax_adf_layer


def relu_adf_layer(params: dict[str, Any], relu_layer: nn.ReLU) -> nn.Module:
    """Convert ReLU deterministic layer to ReLU adf layer."""

    layer = relu_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    relu_adf_layer = layer_fn(**params)
    return relu_adf_layer


def leakyrelu_adf_layer(
    params: dict[str, Any], leakyrelu_layer: nn.LeakyReLU
) -> nn.Module:
    """Convert LeakyReLU deterministic layer to LeakyReLU adf layer."""

    layer = leakyrelu_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    leakyrelu_adf_layer = layer_fn(
        negative_slope=leakyrelu_layer.negative_slope, **params
    )
    return leakyrelu_adf_layer


def dropout_adf_layer(params: dict[str, Any], dropout_layer: nn.Dropout) -> nn.Module:
    """Convert Dropout deterministic layer to Dropout adf layer."""

    layer = dropout_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    dropout_adf_layer = layer_fn(p=dropout_layer.p, **params)
    return dropout_adf_layer


def maxpool2d_adf_layer(
    params: dict[str, Any], maxpool2d_layer: nn.MaxPool2d
) -> nn.Module:
    """Convert maxpool2d deterministic layer to maxpool2d adf layer."""

    layer = maxpool2d_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    maxpool2d_adf_layer = layer_fn(**params)
    return maxpool2d_adf_layer


def avgpool2d_adf_layer(
    params: dict[str, Any], avgpool2d_layer: nn.AvgPool2d
) -> nn.Module:
    """Convert avgpool2d deterministic layer to avgpool2d adf layer."""

    layer = avgpool2d_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    avgpool2d_adf_layer = layer_fn(**params)
    return avgpool2d_adf_layer


def batchnorm2d_adf_layer(
    params: dict[str, Any], batchnorm2d_layer: nn.BatchNorm2d
) -> nn.Module:
    """Convert  batchnorm2d deterministic layer to  batchnorm2d adf layer."""

    layer = batchnorm2d_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    batchnorm2d_adf_layer = layer_fn(
        num_features=batchnorm2d_layer.num_features,
        eps=batchnorm2d_layer.eps,
        momentum=batchnorm2d_layer.momentum,
        affine=batchnorm2d_layer.affine,
        track_running_stats=batchnorm2d_layer.track_running_stats,
        **params,
    )
    return batchnorm2d_adf_layer


def conv_adf_layer(
    params: dict[str, Any], conv_layer: Union[nn.Conv2d, nn.ConvTranspose2d]
) -> nn.Module:
    """Convert conv deterministic layer to conv adf layer."""

    layer = conv_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    linear_adf_layer = layer_fn(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        output_padding=conv_layer.output_padding,
        groups=conv_layer.groups,
        bias=conv_layer.bias,
        dilation=conv_layer.dilation,
        padding_mode=conv_layer.padding_mode,
        **params,
    )
    return linear_adf_layer


def linear_adf_layer(params: dict[str, Any], linear_layer: nn.Linear) -> nn.Module:
    """Convert linear deterministic layer to linear adf layer."""

    layer = linear_layer.__class__.__name__ + "_adf"
    layer_fn = getattr(adf_layers, layer)
    linear_adf_layer = layer_fn(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        **params,
    )
    return linear_adf_layer


def convert_deterministic_to_adf(deterministic_model: nn.Module) -> None:
    """Replace layers with adf layers.

    Args:
        deterministic_model: nn.module
    """
    module_names = deterministic_model.named_children()
    for name, _ in module_names:
        layer_names = name.split(".")
        current_module = deterministic_model
        for l_name in layer_names[:-1]:
            current_module = dict(current_module.named_children())[l_name]

        target_layer_name = layer_names[-1]
        current_layer = dict(current_module.named_children())[target_layer_name]

        if "Conv" in current_layer.__class__.__name__:
            setattr(current_module, target_layer_name, conv_adf_layer(current_layer))
        elif "Linear" in current_layer.__class__.__name__:
            setattr(current_module, target_layer_name, linear_adf_layer(current_layer))
        elif "BatchNorm2d" in current_layer.__class__.__name__:
            setattr(
                current_module, target_layer_name, batchnorm2d_adf_layer(current_layer)
            )
        elif "AvgPool2d" in current_layer.__class__.__name__:
            setattr(
                current_module, target_layer_name, avgpool2d_adf_layer(current_layer)
            )
        elif "MaxPool2d" in current_layer.__class__.__name__:
            setattr(
                current_module, target_layer_name, maxpool2d_adf_layer(current_layer)
            )
        elif "Dropout" in current_layer.__class__.__name__:
            setattr(current_module, target_layer_name, dropout_adf_layer(current_layer))
        elif "LeakyReLU" in current_layer.__class__.__name__:
            setattr(
                current_module, target_layer_name, leakyrelu_adf_layer(current_layer)
            )
        elif "ReLU" in current_layer.__class__.__name__:
            setattr(current_module, target_layer_name, relu_adf_layer(current_layer))
        elif "Softmax" in current_layer.__class__.__name__:
            setattr(current_module, target_layer_name, softmax_adf_layer(current_layer))
        else:
            raise ValueError("Module not available as ADF version.")
