"""Simple MLP for Toy Regression Problems."""

from typing import List

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron for regression predictions."""

    def __init__(
        self,
        dropout_p: float = 0.0,
        n_inputs: int = 1,
        n_hidden: List[int] = [100],
        n_outputs: int = 1,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        """Initialize a new instance of MLP.

        Args:
          dropout_p: dropout percentage
          n_inputs: size of input dimension
          n_hidden: list of hidden layer sizes
          n_outputs: number of model outputs
          predict_sigma: whether the model intends to predict sigma term
            when minimizing NLL
          activation_fn: what nonlinearity to include in the network
        """
        super().__init__()
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
        layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        self.model = nn.Sequential(*layers)
        self.n_outputs = n_outputs

    def forward(self, x) -> Tensor:
        """Forward pass through the neural network.

        Args:
          x: input vector to NN of dimension [batch_size, n_inputs]

        Returs:
          output from neural net of dimension [batch_size, n_outputs]
        """
        return self.model(x)
