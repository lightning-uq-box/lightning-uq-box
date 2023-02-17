"""Simple MLP for Toy Regression Problems."""

from typing import List

import torch
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
        predict_sigma: bool = False,
        activation_fn: nn.Module = nn.Tanh(),
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
        layer_sizes = [n_inputs] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [
                nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]),
                activation_fn,
                nn.Dropout(dropout_p) if idx != 1 else nn.Identity(),
            ]
        layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        self.net = nn.Sequential(*layers)
        self.predict_sigma = predict_sigma

    def forward(self, x) -> Tensor:
        """Forward pass through the neural network.

        Args:
          x: input vector to NN of dimension [batch_size, n_inputs]

        Returs:
          output from neural net of dimension [batch_size, n_outputs]
        """
        out = self.net(x)  # batch_size x (mu,sigma)
        # make sure output sigma is always positive
        if self.predict_sigma:
            out[:, 1] = torch.log(1 + torch.exp(out[:, 1])) + 1e-06
        return out
