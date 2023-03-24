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
          preds: batch_size x 2, consisting of mu and log_sigma_2
          target: batch_size x 1, regression targets

        Returns:
          computed loss for the entire batch
        """
        mu, log_sigma_2 = preds[:, 0].unsqueeze(-1), preds[:, 1].unsqueeze(-1)
        loss = 0.5 * log_sigma_2 + (
            0.5 * torch.exp(-log_sigma_2) * torch.pow((target - mu), 2)
        )
        loss = torch.mean(loss, dim=0)
        return loss


class TheirNLL(nn.Module):
    """NLL Loss from Wilson papers.

    https://github.com/wjmaddox/drbayes/blob/0c0c32edade51f1ec471753b7bf258f40bf8fdd6/subspace_inference/losses.py#L4

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a new instance."""
        super().__init__(*args, **kwargs)

    def forward(self, preds: Tensor, target: Tensor):
        """Compute loss."""
        mean = preds[:, 0].view_as(target)
        var = preds[:, 1].view_as(target)

        mse = torch.nn.functional.mse_loss(mean, target, reduction="none")
        mean_portion = mse / (2 * var)
        var_portion = 0.5 * torch.log(var)
        loss = mean_portion + var_portion
        return loss.mean()


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
