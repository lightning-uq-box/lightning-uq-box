# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# adapted from https://github.com/y0ast/DUE/blob/main/due/fc_resnet.py

"""Fully Connected ResNet architecture."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FCResNet(nn.Module):
    """Fully Connected ResNet architecture.

    ResFNN architecture

    Introduced in SNGP: https://arxiv.org/abs/2006.10108
    """

    def __init__(
        self,
        input_dim: int,
        features: int,
        depth: int,
        dropout_rate: float = 0.01,
        num_outputs: int = None,
        activation: str = "relu",
    ):
        """Initialze a new instance of the FCResNet.

        Args:
            input_dim: The input dimension of the network
            features: The number of features in the hidden layers
            depth: The number of hidden layers
            dropout_rate: The dropout rate
            num_outputs: The number of outputs
            activation: The activation function, "elu" or "relu"

        Raises:
            ValueError: If the activation is not known
        """
        super().__init__()
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

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
