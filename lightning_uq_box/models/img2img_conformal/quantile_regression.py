# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

import torch.nn as nn
from torch import Tensor
import torch

class QuantileRegressionLayer(nn.Module):
    def __init__(self, n_channels_middle: int, n_channels_out: int, q_lo: float, q_hi: float):
        """
        Initialize the QuantileRegressionLayer class.

        Args:
            n_channels_middle: The number of middle channels.
            n_channels_out: The number of output channels.
        """
        super(QuantileRegressionLayer, self).__init__()
        self.q_lo = q_lo
        self.q_hi = q_hi

        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the QuantileRegressionLayer.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        output = torch.cat((self.lower(x).unsqueeze(1), self.prediction(x).unsqueeze(1), self.upper(x).unsqueeze(1)), dim=1)
        return output
    


class PinballLoss:
    # TODO can we use our pinball loss already
    def __init__(self, quantile: float = 0.10, reduction: str = 'mean'):
        """
        Initialize the PinballLoss class.

        Args:
            quantile: The quantile for the pinball loss. Default is 0.10.
            reduction: The reduction method to apply ('mean' or 'sum'). Default is 'mean'.
        """
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Compute the pinball loss.

        Args:
            output: The output tensor.
            target: The target tensor.

        Returns:
            The computed pinball loss.
        """
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
    

class QuantileLoss(nn.Module):
    def __init__(self, q_lo: float, q_hi: float, q_lo_weight: float, q_hi_weight: float, mse_weight: float):
        """
        Initialize the custom loss function.

        Args:
            q_lo: The lower quantile.
            q_hi: The upper quantile.
            q_lo_weight: The weight for the lower quantile loss.
            q_hi_weight: The weight for the upper quantile loss.
            mse_weight: The weight for the mean squared error loss.
        """
        super().__init__()
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.q_lo_weight = q_lo_weight
        self.q_hi_weight = q_hi_weight
        self.mse_weight = mse_weight
        self.q_lo_loss = PinballLoss(quantile=self.q_lo)
        self.q_hi_loss = PinballLoss(quantile=self.q_hi)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the custom loss.

        Args:
            pred: The prediction tensor.
            target: The target tensor.

        Returns:
            The computed loss.
        """
        loss = self.q_lo_weight * self.q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
            self.q_hi_weight * self.q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
            self.mse_weight * self.mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze())
        return loss