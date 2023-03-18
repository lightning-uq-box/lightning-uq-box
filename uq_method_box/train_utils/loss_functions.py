"""Loss Functions specific to UQ-methods."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class NLL(nn.Module):
    """Negative Log Likelihood loss."""

    def __init__(self):
        """Initialize a new instance of NLL."""
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor):
        """Compute NLL Loss.

        Args:
          preds: batch_size x 2, consisting of mu and sigma
          target: batch_size x 1, regression targets

        Returns:
          computed loss for the entire batch
        """
        eps = torch.ones_like(target) * 1e-6
        mu, sigma = preds[:, 0].unsqueeze(-1), preds[:, 1].unsqueeze(-1)
        loss = torch.log(sigma**2) + ((target - mu) ** 2 / torch.max(sigma**2, eps))
        loss = torch.mean(loss, dim=0)
        return loss


class QuantileLoss(nn.Module):
    """Quantile or Pinball Loss function."""

    def __init__(self, quantiles: List[float]) -> None:
        """Initialize a new instance of Quantile Loss function."""
        super().__init__()
        self.quantiles = quantiles

    def pinball_loss(self, y: Tensor, y_hat: Tensor, alpha: float):
        """Pinball Loss for a desired quantile alpha.

        Args:
            y: true targets of shape [batch_size]
            y_hat: model predictions for specific quantile of shape [batch_size]
            alpha: quantile

        Returns:
            computed loss for a single quantile
        """
        delta = y - y_hat  # (shape: (batch_size))
        abs_delta = torch.abs(delta)  # (shape: (batch_size))

        loss = torch.zeros_like(y)  # (shape: (batch_size))
        loss[delta > 0] = alpha * abs_delta[delta > 0]  # (shape: (batch_size))
        loss[delta <= 0] = (1.0 - alpha) * abs_delta[
            delta <= 0
        ]  # (shape: (batch_size))
        loss = torch.mean(loss)
        return loss

    def forward(self, preds: Tensor, target: Tensor):
        """Compute Quantile Loss.

        Args:
            preds: model predictions of shape [batch_size x num_quantiles]
            target: targets of shape [batch_size x 1]

        Returns:
            computed loss for all quantiles over the entire batch
        """
        loss = torch.stack(
            [
                self.pinball_loss(preds[:, idx], target.squeeze(), alpha)
                for idx, alpha in enumerate(self.quantiles)
            ]
        )
        return loss.mean()
