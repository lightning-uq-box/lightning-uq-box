# MIT License
# Copyright (c) 2021 Nikita Durasov

# Implementation adapted from above
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.


"""Masked Ensemble Layers.

Adapted from https://github.com/nikitadurasov/masksembles
to be integrated with Lightning and ported to PyTorch.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.common_types import _size_2_t


class MasksemblesLayer(nn.Module):
    """Mask Ensemble Operation Layer.

    If you are using Masksembles in your model, please cite:

    * https://arxiv.org/abs/2012.08334
    """

    def __init__(self, channels: int, num_masks: int, scale: float) -> None:
        """Initialize the Masksembles operation layer.

        Args:
            channels: The number of channels in the input tensor.
            num_masks: The number of masks to generate.
            scale: The scale factor for mask generation. Should be
                be a scaler in the interval [1, 6]

        Raises:
            AssertionError: If the scale factor is less than 1.
        """
        super().__init__()

        self.channels = channels
        self.num_masks = num_masks

        assert (
            scale >= 1 and scale <= 6
        ), "Scale factor should be in the interval [1, 6]."

        self.scale = scale

        masks = generation_wrapper(channels, num_masks, scale)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).float()

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies the masksembles technique to the input tensor.

        Parameters:
            inputs: The input tensor to be masked

        Returns:
            The output tensor after applying masksembles.

        Raises:
            ValueError: If the input tensor is not 2D or 4D.
        """
        x = rearrange(inputs, "(n b) ... -> n b ...", n=self.num_masks)
        if inputs.dim() == 2:  # Assuming (batch, num_features) for vector data
            x = x * self.masks.unsqueeze(1)
        elif inputs.dim() == 4:  # Assuming (batch, channel, height, width) for images
            x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(
                "Masksembles approach is only implemented for vector (batch_size, num_features) or image data (batch_size, channels, height, width)"
            )
        return rearrange(x, "n b ... -> (n b) ...")


def generate_masks_(m: int, num_masks: int, s: float) -> Tensor:
    """Generates binary masks for the Masksembles layer.

    Parameters:
        m: The number of features to mask.
        n: The number of masks (subnetworks) to generate.
        s: The scale factor determining the sparsity of the masks.

    Returns:
        A tensor containing the generated binary masks.
    """
    total_positions = int(m * s)
    masks_list = []

    for _ in range(num_masks):
        new_vector = torch.zeros(total_positions)
        idx = torch.randperm(total_positions)[:m]
        new_vector[idx] = 1
        masks_list.append(new_vector)

    masks = torch.stack(masks_list)
    # drop useless positions
    masks = masks[:, ~(masks == 0).all(dim=0)]
    return masks


def generate_masks(m: int, num_masks: int, s: float) -> Tensor:
    """Generates binary masks for the Masksembles layer.

    This function wraps `generate_masks_` to ensure the generated masks meet the
    expected size criteria derived from the parameters.

    Parameters:
        m: The number of features to mask.
        n: The number of masks (subnetworks) to generate.
        s: The scale factor determining the sparsity of the masks.

    Returns:
        A tensor containing the generated binary masks.
    """
    masks = generate_masks_(m, num_masks, s)
    expected_size = int(m * s * (1 - (1 - 1 / s) ** num_masks))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, num_masks, s)
    return masks


def generation_wrapper(c: int, num_masks: int, scale: float) -> Tensor:
    """A wrapper function for generating masks.

    This function iteratively adjusts the scale parameter to generate masks
    that match the desired number of features exactly.

    Parameters:
        c: The desired number of features in the masks.
        n: The number of masks (subnetworks) to generate.
        scale: The initial scale factor for mask generation.

    Returns:
        A tensor containing the generated binary masks.

    Raises:
        AssertionError: If the number of features is less than 10.
        AssertionError: If the scale factor is not in the interval [1, 6].
        ValueError: If the desired number of features cannot be achieved with
                    the given parameters.
    """
    assert c >= 10, "Number of features must be >= 10."
    assert scale >= 1 and scale <= 6, "Scale factor should be in the interval [1, 6]."

    active_features = int(c / (scale * (1 - (1 - 1 / scale) ** num_masks)))

    max_iter = 1000
    min_scale = max(scale * 0.8, 1.0)
    max_scale = scale * 1.2

    for _ in range(max_iter):
        mid_scale = (min_scale + max_scale) / 2
        masks = generate_masks(active_features, num_masks, mid_scale)
        if masks.shape[-1] == c:
            break
        elif masks.shape[-1] > c:
            max_scale = mid_scale
        else:
            min_scale = mid_scale

    if masks.shape[-1] != c:
        raise ValueError(
            "generation_wrapper function failed to generate masks with requested number of features. Please try to change scale parameter"
        )

    return masks


class MaskedLinear(nn.Module):
    """Masked Linear Layer.

    If you are using Masksembles in your model, please cite:

    * https://arxiv.org/abs/2012.08334
    """

    def __init__(
        self,
        num_estimators: int,
        scale: float,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize a new instance of MaskedLinear.

        Args:
            num_estimators: The number of estimators (masks) to generate.
            scale: The scale factor for mask generation. Muste be a scaler in
                the interval [1, 6].
            in_features: The number of input features to the Linear layer.
            out_features: The number of output features of the Linear layer.
            bias: Whether to include a bias term in the linear transformation.

        Raises:
            AssertionError: If the scale factor is not in the interval [1, 6].
            AssertionError: If the number of input features is less than 10.
        """
        super().__init__()

        assert (
            scale >= 1 and scale <= 6
        ), "Scale factor should be in the interval [1, 6]."
        assert in_features >= 10, "Number of input features must be >= 10."

        self.mask = MasksemblesLayer(in_features, num_masks=num_estimators, scale=scale)
        self.in_features = in_features
        self.out_features = out_features
        self.num_estimators = num_estimators
        self.scale = scale
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies the MaskedLinear forward pass to the input tensor.

        Args:
            inputs: The input tensor to be transformed

        Returns:
            The output tensor after applying the MaskedLinear transformation.
        """
        return self.linear(self.mask(inputs))

    def extra_repr(self):
        """Representation when printing out layer."""
        return (
            f"num_estimators={self.num_estimators}, scale={self.scale},"
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"
        )


class MaskedConv2d(nn.Module):
    """Masked Conv2d Layer.

    If you are using Masksembles in your model, please cite:

    * https://arxiv.org/abs/2012.08334
    """

    def __init__(
        self,
        num_estimators: int,
        scale: float,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize a new instance of MaskedConv2d.

        Args:
            num_estimators: The number of estimators (masks) to generate.
            scale: The scale factor for mask generation. Muste be a scaler in
                the interval [1, 6].
            in_channels: The number of input channels to the Conv2d layer.
            out_channels: The number of output channels of the Conv2d layer.
            kernel_size: The size of the convolutional kernel.
            stride: The stride of the convolution operation.
            padding: The padding to apply to the input tensor.
            dilation: The dilation rate of the convolution operation.
            groups: The number of groups that connect inputs and outputs.
            bias: Whether to include a bias term in the convolution operation.

        Raises:
            AssertionError: If the scale factor is not in the interval [1, 6].
            AssertionError: If the number of input channels is less than 10.
        """
        super().__init__()

        assert (
            scale >= 1 and scale <= 6
        ), "Scale factor should be in the interval [1, 6]."
        assert in_channels >= 10, "Number of input features must be >= 10."

        self.mask = MasksemblesLayer(in_channels, num_masks=num_estimators, scale=scale)

        self.num_estimators = num_estimators
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies the MaskedConv2d forward pass to the input tensor.

        Args:
            inputs: The input tensor to be transformed

        Returns:
            The output tensor after applying the MaskedConv2d transformation.
        """
        return self.conv_layer(self.mask(inputs))

    def extra_repr(self):
        """Representation when printing out layer."""
        return (
            f"num_estimators={self.num_estimators}, scale={self.scale},"
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias}"
        )
