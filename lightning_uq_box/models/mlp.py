# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Simple MLP for Toy Problems."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """Multi-layer perceptron for predictions."""

    def __init__(
        self,
        dropout_p: float = 0.0,
        n_inputs: int = 1,
        n_hidden: list[int] = [100],
        n_outputs: int = 1,
        activation_fn: nn.Module | None = None,
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
        if activation_fn is None:
            activation_fn = nn.ReLU()
        layer_sizes = [n_inputs] + n_hidden
        layers = []
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


class IterMLP(nn.Module):
    """Iterative Uncertainty MLP."""

    def __init__(self, n_inputs: int = 1, n_outputs: int = 2):
        """Initialize a new instance of net."""
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, n_outputs)
        self.activation = nn.ELU()

        self.cf = nn.Linear(n_outputs, 20)

        self.num_inputs = n_inputs
        self.num_outputs = n_outputs

    def forward(self, x: Tensor, pred_prob: Tensor | None = None):
        """Forward pass of Iterative NN.

        Args:
            x: input tensor.
            pred_prob: predicted logit values
                of iterative UQ

        Returns:
            logits network output
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        # iterative part, if "pred_prob"
        # is not None, we use it for inference
        if pred_prob is not None:
            pred_prob = torch.nn.functional.softmax(pred_prob, 1)
            x += self.activation(self.cf(pred_prob))

        return self.fc3(x)


class IterCNN(nn.Module):
    """Iterative Uncertainty CNN."""

    def __init__(self, num_inputs: int = 1, num_outputs: int = 2):
        """Initialize a new instance of net."""
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, num_outputs)
        self.activation = nn.ELU()

        self.cf = nn.Linear(10, 50)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, x: Tensor, pred_prob: Tensor | None = None):
        """Forward pass of Iterative NN.

        Args:
            x: input tensor.
            pred_prob: predicted logit values
                of iterative UQ

        Returns:
            logits network output
        """
        x = self.activation(F.max_pool2d(self.conv1(x), 2))
        x = self.activation(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.activation(self.fc1(x))

        # iterative part, if "pred_prob"
        # is not None, we use it for inference
        if pred_prob is not None:
            pred_prob = torch.nn.functional.softmax(pred_prob, 1)
            x += self.activation(self.cf(pred_prob))

        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
