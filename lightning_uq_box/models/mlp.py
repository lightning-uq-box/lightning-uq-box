# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Simple MLP for Toy Problems."""

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron for predictions."""

    def __init__(
        self,
        dropout_p: float = 0.0,
        n_inputs: int = 1,
        n_hidden: list[int] = [100],
        n_outputs: int = 1,
        n_targets: int = 1,
        activation_fn: nn.Module | None = None,
    ) -> None:
        """Initialize a new instance of MLP.

        Args:
          dropout_p: dropout percentage
          n_inputs: size of input dimension
          n_hidden: list of hidden layer sizes
          n_outputs: number of model outputs per target
          n_targets: number of targets to predict, for
            1D regression this is 1, for multivariate regression
            this is the number of regression targets
          activation_fn: what nonlinearity to include in the network
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_targets = n_targets

        if activation_fn is None:
            activation_fn = nn.ReLU()
        layers = []
        # layer sizes
        layer_sizes = [n_inputs] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [
                nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]),
                activation_fn,
                nn.Dropout(dropout_p),  # if idx != 1 else nn.Identity(),
            ]
        # add output layer
        layers += [nn.Linear(layer_sizes[-1], n_outputs * n_targets)]
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        """Forward pass through the neural network.

        Args:
          x: input vector to NN of dimension [batch_size, n_inputs]

        Returs:
          output from neural net of dimension [batch_size, n_outputs] if num_targets=1
          else [batch_size, n_outputs, num_targets]
        """
        if self.n_targets > 1:
            return self.model(x).view(-1, self.n_outputs, self.n_targets)
        else:
            return self.model(x)
