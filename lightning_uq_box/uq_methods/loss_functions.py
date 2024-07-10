# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Loss Functions specific to UQ-methods."""

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
            pred_losses: nll of predictions vs targets [num_samples, batch_size, 1]
            log_f_hat: ["num_samples"]
            log_Z_prior: 0 shape
            log_normalizer: 0 shape
            log_normalizer_z: 0 shape
            log_f_hat_z: [num_samples,batch_size]
            loss_terms: collected loss terms over the variational layer weights
            #where are the loss terms?
            N: number of datapoints in dataset #train data set?
            alpha: alpha divergence value

        Returns:
            energy function loss
        """
        S = pred_losses.size(dim=1)
        n_samples = pred_losses.size(dim=0)
        alpha = torch.tensor(self.alpha).to(pred_losses.device)
        NoverS = torch.tensor(self.N / S).to(pred_losses.device)
        one_over_n_samples = torch.tensor(1 / n_samples).to(pred_losses.device)
        one_over_N = torch.tensor(1 / self.N).to(pred_losses.device)

        # if we change y_pred: Normal dist output
        # with shape [batch_size, num_samples, output_dim]
        # to be a Normal dist output
        # with shape [num_samples, batch_size, output_dim]
        # then the below should be y_pred.log_prob(y[None, :, :]).sum(-1)?
        # Can we do this to be lazy and
        # avoid the changing of forward passes in each layer?
        # the outer torch.sum need to be taken over the batchsize
        # the inner logsumexp over the numer of samples
        inner_term = self.alpha * (
            -pred_losses.sum(-1) - (1 / self.N) * log_f_hat[:, None] - log_f_hat_z
        )
        loss = (
            -(1 / alpha)
            * NoverS
            * torch.sum(
                torch.logsumexp(
                    inner_term,
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
        return loss * one_over_N


# TODO to be addded for pixel wise regression
# class LowRankMultivariateNormal_NLL(nn.Module):
#     """Negative Log Likelihood loss."""

#     def __init__(self, rank=10, eps=1e-8):
#         """Initialize a new instance of LowRankMultivariateNormal_NLL.

#         Args:
#           rank: rank (=number of columns) of covariance matrix factor matrix.
#           eps: eps-value for strictly positive diagonal Psi

#         """
#         super().__init__()
#         self.rank = rank
#         self.eps = eps

#     def forward(self, preds: Tensor, target: Tensor):
#         """Compute LowRankMultivariateNormal_NLL Loss.

#         Args:
#           preds: batch_size x (rank + 2) x target_shape, consisting of
#                mu,Gamma and Psi
#           target: batch_size x target_shape, regression targets

#         Returns:
#           computed loss for the entire batch
#         """

#         mu, gamma, psi = (
#             preds[:, 0:1],
#             preds[:, 1 : self.rank + 1],
#             preds[:, self.rank + 1].exp() + self.eps,
#         )

#         [b, w, h] = target.shape

#         gamma = gamma.reshape([b, w * h, self.rank])
#         psi = torch.diag(psi, diagonal=-1)

#         lowrank_norm = torch.distributions.LowRankMultivariateNormal(
#             loc=mu, cov_factor=gamma, cov_diag=psi
#         )

#         loss = -lowrank_norm.log_prob(target)
#         loss = torch.mean(loss, dim=0)
#         return loss


class NLL(nn.Module):
    """Negative Log Likelihood loss."""

    def __init__(self):
        """Initialize a new instance of NLL."""
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor):
        """Compute NLL Loss.

        Args:
          preds: batch_size x 2 x other dims, consisting of mu and log_sigma_2
          target: batch_size x 1 x other dims, regression targets

        Returns:
          computed loss for the entire batch
        """
        mu, log_sigma_2 = preds[:, 0:1, ...], preds[:, 1:2, ...]
        loss = 0.5 * log_sigma_2 + (
            0.5 * torch.exp(-log_sigma_2) * torch.pow((target - mu), 2)
        )
        return loss.mean()


class PinballLoss(nn.Module):
    """Pinball Loss for quantile regression."""

    def __init__(self, quantiles: list[float]):
        """Initialize a new instance of Pinball Loss.

        Args:
            quantiles: List of quantiles for which to compute the loss.
        """
        super().__init__()
        self.quantiles = quantiles

    def pinball_loss(self, y_pred: Tensor, y_true: Tensor, tau: float) -> Tensor:
        """Compute the Pinball Loss for a single quantile.

        Args:
            y_pred: Predicted values.
            y_true: True values.
            tau: The quantile for which to compute the loss.

        Returns:
            The computed Pinball Loss for the given quantile.
        """
        err = y_true - y_pred
        loss = torch.where(err >= 0, tau * err, (tau - 1) * err)
        return torch.mean(loss)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute the Pinball Loss for all quantiles.

        Args:
            preds: Predicted values for all quantiles.
            target: True values.

        Returns:
            The mean Pinball Loss for all quantiles.
        """
        loss = torch.stack(
            [
                self.pinball_loss(preds[:, idx], target.squeeze(), tau)
                for idx, tau in enumerate(self.quantiles)
            ]
        )
        return loss.mean()


class DERLoss(nn.Module):
    """Deep Evidential Regression Loss.

    Taken from `here <https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py#L61>`_. # noqa: E501

    This implements the loss corresponding to equation 12
    from the `paper <https://arxiv.org/abs/2205.10060>`_.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2205.10060
    """

    def __init__(self, coeff: float = 0.01) -> None:
        """Initialize a new instance of the loss function.

        Args:
          coeff: loss function coefficient
        """
        super().__init__()
        self.coeff = coeff

    def forward(self, logits: Tensor, y_true: Tensor):
        """DER Loss.

        Args:
          logits: predicted tensor from model [batch_size x 4 x other dims]
          y_true: true regression target of shape [batch_size x 1 x other dims]

        Returns:
          DER loss
        """
        assert (
            logits.shape[1] == 4
        ), "logits should have shape [batch_size x 4 x other dims]"
        assert (
            y_true.shape[1] == 1
        ), "y_true should have shape [batch_size x 1 x other dims]"
        gamma, nu, _, beta = (
            logits[:, 0:1, ...],
            logits[:, 1:2, ...],
            logits[:, 2:3, ...],
            logits[:, 3:4, ...],
        )
        error = gamma - y_true
        var = beta / nu

        return torch.mean(torch.log(var) + (1.0 + self.coeff * nu) * error**2 / var)


class VAELoss(nn.Module):
    """VAE Loss Function.
    
    Consists of the KL Divergence and Reconstruction Loss (MSE).
    """

    def __init__(self, kl_scale: float = 1e-4) -> None:
        """Initialize a new instance of the VAE Loss Function.

        Args:
            kl_scale: The scale of the KL loss. Default is 1e-4, based on empirical
                results. However, this value can significantly affect the
                reconstruction and sample diversity.
        """
        super().__init__()
        self.kl_scale = kl_scale

    def forward(
        self, x_recon: Tensor, target: Tensor, mu: Tensor, log_var: Tensor
    ) -> tuple[Tensor]:
        """Compute the VAE Loss.

        Args:
            x_recon: The reconstructed input tensor.
            target: The input tensor.
            mu: The mean of the latent space.
            log_var: The log variance of the latent space.

        Returns:
            The computed KL and reconstruction loss
        """
        recon_loss = nn.functional.mse_loss(x_recon, target, reduction="mean")
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        return self.kl_scale * KLD, recon_loss
