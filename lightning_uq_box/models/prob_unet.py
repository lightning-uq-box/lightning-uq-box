# Copyright 2019 Stefan Knegt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.
# Changes from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/probabilistic_unet.py: # noqa: E501
# - add doc strings and type hints
# - remove unused intializer args for Encoder and AxisAlignedConvGaussian
# - change fcomb arguments to accomodate lightning module

"""Probabilistic UNet Model parts."""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal


def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> None:
    """Fills the input Tensor with values drawn from a truncated normal distribution.

    Args:
        tensor: Input tensor
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m: nn.Module) -> None:
    """Initialize weights of the model.

    Args:
        m: Model whose weights need to be initialized
    """
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias:
            truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m: nn.Module) -> None:
    """Initialize weights of the model with orthogonal initialization.

    Args:
        m: Model whose weights need to be initialized
    """
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        if m.bias:
            truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


class Encoder(nn.Module):
    """Encoder Cpnvolutional Net for The AxisAlignedConvGaussian.

    A convolutional neural network, consisting of len(num_filters) times
    a block of no_convs_per_block convolutional layers,
    After each block a pooling operation is performed.
    And after each convolutional layer a RELU is applied.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        padding: bool = True,
        posterior: bool = False,
    ) -> None:
        """Initialize a new instance of Encoder.

        Args:
            input_channels: Number of input channels
            num_filters: List of number of filters for each layer
            no_convs_per_block: Number of convolutions per block
            padding: Whether to use padding
            posterior: Whether this is a posterior encoder
        """
        super().__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # the mask is concatenated at the channel axis,
            # so increase the input_channels.
            self.input_channels += 1

        layers: list[nn.Module] = []
        output_dim: int = 0
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block.
            The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim: int = self.input_channels if i == 0 else output_dim  # noqa: F821
            output_dim = num_filters[i]

            if i != 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
                )

            layers.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding))
            )
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(
                    nn.Conv2d(
                        output_dim, output_dim, kernel_size=3, padding=int(padding)
                    )
                )
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Encoder network.

        Args:
            input: Input tensor

        Returns:
            Output tensor
        """
        return self.layers(input)


class AxisAlignedConvGaussian(nn.Module):
    """Conv Net that parametrizes a Gaussian with axis aligned cov matrix."""

    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        latent_dim: int,
        posterior: bool = False,
    ) -> None:
        """Initialize a new instance of AxisAlignedConvGaussian.

        Args:
            input_channels: Number of input channels
            num_filters: List of number of filters for each layer
            no_convs_per_block: Number of convolutions per block
            latent_dim: Dimension of the latent space
            initializers: Dictionary of initializers for the layers
            posterior: Whether this is a posterior network or not
        """
        super().__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.encoder = Encoder(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            posterior=self.posterior,
        )
        self.conv_layer = nn.Conv2d(
            num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1
        )
        self.show_img: Optional[Tensor] = None
        self.show_seg: Optional[Tensor] = None
        self.show_concat: Optional[Tensor] = None
        self.sum_input: Optional[Tensor] = None
        self.show_enc: Optional[Tensor] = None

        nn.init.kaiming_normal_(
            self.conv_layer.weight, mode="fan_in", nonlinearity="relu"
        )
        if self.conv_layer.bias is not None:
            nn.init.normal_(self.conv_layer.bias)

    def forward(self, input: Tensor, segm: Optional[Tensor] = None) -> Independent:
        """Forward pass of the AxisAlignedConvGaussian network.

        Args:
            input: Input tensor
            segm: Optional segmentation mask to concatenate to the input

        Returns:
            A multivariate normal distribution with diagonal covariance matrix
        """
        # If segmentation is not none, comcat mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise
        # it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]  # noqa: E203

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """Fcomb network.

    A function composed of no_convs_fcomb times a 1x1 convolution that
    combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along
    their channel axis.
    """

    def __init__(
        self,
        input_size: int,
        filter_size: int,
        num_classes: int,
        no_convs_fcomb: int,
        initializers: Dict[str, Callable],
        use_tile: bool = True,
    ) -> None:
        """Initialize a new instance of Fcomb.

        Args:
            input_size: Number of output features of the UNet + latent dimensions
            filter_size: filter size
            num_classes: Number of classes
            no_convs_fcomb: Number of 1x1 convolutions
            initializers: Dictionary of initializers for the layers
            use_tile: Whether to use tiling
        """
        # TODO combine filter size with no_convs_fcomb as a list because
        # it is redundant
        super().__init__()
        self.input_size = input_size
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = "Fcomb"

        if self.use_tile:
            layers: List[nn.Module] = []

            # Decoder of N x a 1x1 convolution followed by a ReLU except for last layer
            layers.append(nn.Conv2d(self.input_size, self.filter_size, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb - 2):
                layers.append(
                    nn.Conv2d(self.filter_size, self.filter_size, kernel_size=1)
                )
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(
                self.filter_size, self.num_classes, kernel_size=1
            )

            if initializers["w"] == "orthogonal":
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a: Tensor, dim: int, n_tile: int) -> Tensor:
        """Tile function.

        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3

        Args:
            a: Input tensor
            dim: Dimension along which to tile
            n_tile: Number of times to tile

        Returns:
            Tiled tensor
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(next(self.parameters()).device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map: Tensor, z: Tensor) -> Tensor:
        """Forward pass of the Fcomb network.

        Concatenates the feature map with the sampled z from the latent space,
        and passes it through the fcbomb layers.

        Args:
            feature_map: Feature map output of the UNet
            z: Sample taken from the latent space

        Returns:
            Output tensor
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

        # Concatenate the feature map (output of the UNet) and the sample
        # taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return self.last_layer(output)
