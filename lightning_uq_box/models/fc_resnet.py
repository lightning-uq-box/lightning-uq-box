# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

# adapted from https://github.com/y0ast/DUE/blob/main/due/fc_resnet.py

"""Fully Connected ResNet architecture."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lightning_uq_box.uq_methods.spectral_normalized_layers import spectral_norm_fc


class FCResNet(nn.Module):
    """Fully Connected ResNet architecture."""

    def __init__(
        self,
        input_dim: int,
        features: int,
        depth: int,
        spectral_normalization: bool,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        dropout_rate: float = 0.01,
        num_outputs: int = None,
        activation: str = "relu",
    ):
        """Initialze a new instance of the FCResNet.

        Args:
            input_dim: The input dimension of the network.
            features: The number of features in the hidden layers.
            depth: The number of hidden layers.
            spectral_normalization: Whether to use spectral normalization.
            coeff: The coefficient for spectral normalization.
            n_power_iterations: The number of power iterations for spectral
                normalization.
            dropout_rate: The dropout rate.
            num_outputs: The number of outputs.
            activation: The activation function.
        """
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the network.

        Args:
            x: The input tensor

        Returns:
            The output tensor
        """
        x = self.first(x)

        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x)

        return x
