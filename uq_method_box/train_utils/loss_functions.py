"""Loss Functions specific to UQ-methods."""

import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class EnergyAlphaDivergence(nn.Module):
    """Energy Alpha Divergence Loss."""

    def __init__(self, N: int, alpha: float = 1.0) -> None:
        """Initialize a new instance of Energy Alpha Divergence loss.

        Args:
            N: number of datapoints in training set
            alpha: alpha divergence parameter
        """
        super().__init__()
        self.N = N
        self.alpha = alpha

    def forward(
        self,
        pred_losses: Tensor,
        log_f_hat: Tensor,
        log_Z_prior: Tensor,
        log_normalizer: Tensor,
        log_normalizer_z: Tensor,
        log_f_hat_z: Tensor,
    ) -> Tensor:
        """Compute the energy function loss.

        Args:
            pred_losses: [batch_size, num_samples, 1]
            log_f_hat: ["num_samples"]
            log_Z_prior: 0 shape
            log_normalizer: 0 shape
            log_noramlizer_z: 0 shape
            log_f_hat_z: 0 shape
            loss_terms: collected loss terms over the variational layer weights
            #where are the loss terms?
            N: number of datapoints in dataset #train data set?
            alpha: alpha divergence value

        Returns:
            energy function loss
        """
        S = pred_losses.size(dim=1)
        n_samples = pred_losses.size(dim=0)
        alpha = torch.tensor(self.alpha)
        NoverS = torch.tensor(self.N / S)
        one_over_n_samples = torch.tensor(1 / n_samples)
        # if we change y_pred: Normal dist output
        # with shape [batch_size, num_samples, output_dim]
        # to be a Normal dist output
        # with shape [num_samples, batch_size, output_dim]
        # then the below should be y_pred.log_prob(y[None, :, :]).sum(-1)?
        # Can we do this to be lazy and
        # avoid the changing of forward passes in each layer?
        # the outer torch.sum need to be taken over the batchsize
        # the inner logsumexp over the numer of samples
        return (
            -(1 / alpha)
            * NoverS
            * torch.sum(
                torch.logsumexp(
                    self.alpha * (pred_losses - (1 / self.N) * log_f_hat - log_f_hat_z),
                    dim=0,  # number of dimensions that should be reduced
                    # i.e. logsumexp over samples
                )
                + torch.log(one_over_n_samples),
                dim=0,
            )
            - log_normalizer
            - NoverS * log_normalizer_z
            + log_Z_prior
        )


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
