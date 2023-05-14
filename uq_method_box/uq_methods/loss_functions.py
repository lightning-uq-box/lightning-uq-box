"""Loss Functions specific to UQ-methods."""

import math

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


class QuantileLoss(nn.Module):
    """Quantile or Pinball Loss function."""

    def __init__(self, quantiles: list[float]) -> None:
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


class DERLoss(nn.Module):
    """Deep Evidential Regression Loss.

    Taken from: https://github.com/pasteurlabs/unreasonable_effective_der/blob/
    4631afcde895bdc7d0927b2682224f9a8a181b2c/models.py#L46

    This implements the loss corresponding to equation ...

    """

    def __init__(self, coeff: float = 0.01) -> None:
        """Initialize a new instance of the loss function.

        Args:
          coeff: loss function coefficient
        """
        super().__init__()
        self.coeff = coeff

    def forward(self, y_pred: Tensor, y_true: Tensor):
        """DER Loss.

        Args:
          y_pred: predicted tensor from model [batch_size x 4]
          y_true: true regression target of shape [batch_size x 1]

        Returns:
          DER loss
        """
        y_true = y_true.squeeze(-1)
        gamma, nu, alpha, beta = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
        error = gamma - y_true
        omega = 2.0 * beta * (1.0 + nu)

        return torch.mean(
            0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(error**2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
            + self.coeff * torch.abs(error) * (2.0 * nu + alpha)
        )
