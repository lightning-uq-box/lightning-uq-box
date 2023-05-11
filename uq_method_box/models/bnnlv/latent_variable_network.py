"""Latent Variable Network."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers.utils import calc_log_f_hat, calc_log_normalizer


class LatentVariableNetwork(nn.Module):
    """Latent Variable Network for BNN+LV."""

    def __init__(
        self,
        net: nn.Module,
        num_training_points: int,
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 1.0,
        lv_init_mu: float = 0.0,
        lv_init_std: float = 1.0,
        lv_latent_dim: int = 1,
        n_samples: int = 25,
    ) -> None:
        """Initialize a new Latent Variable Network.

        Args:
            net:
            num_training_points:
            lv_prior_mu:
            lv_prior_std:
            lv_init_mu:
            lv_init_std:
            lv_latent_dim:
            n_samples:
        """
        super().__init__()

        self.net = net
        self.num_training_points = num_training_points
        self.lv_prior_mu = lv_prior_mu
        self.lv_prior_std = lv_prior_std
        self.lv_init_mu = lv_init_mu
        self.lv_init_std = lv_init_std
        self.lv_latent_dim = lv_latent_dim
        self.n_samples = n_samples

        self.log_var_init = np.log(np.expm1(lv_init_std))

        self.z_mu = torch.tensor(0.0)
        self.z_log_sigma = torch.tensor(np.log(np.expm1(self.lv_init_std)))

        self.log_f_hat_z = None
        self.log_normalizer_z = None
        self.weight_eps = None

    def fix_randomness(self) -> None:
        """Fix the randomness of reparameterization trick."""
        # weight eps vector large enough to cover entire
        # dataset for full batch theoretically
        self.weight_eps = torch.randn_like(
            self.num_training_points, self.lv_latent_dim
        ).to(self.device)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass where latent vector is sampled.

        Args:
            x: input X [batch_size, num_features]
            y: target y [batch_size, target_dim]

        Returns:
            z vector of dimension [batch_size, latent_dim]
        """
        if self.weight_eps is None:
            weights_eps = torch.randn(x.shape[0], self.lv_latent_dim).to(x.device)
        else:
            # always extract the same weight_eps since we
            # want it fixed
            weights_eps = self.weight_eps[: x.shape[0], :]

        # pass through NN
        x = self.net(torch.cat([x, y], dim=-1))  # out [batch_size, lv_latent_dim*2]

        # make sure q(z) is close to N(0,1) at initialization
        init_scaling = 0.1
        # extract encoded mean from NN output
        z_mu = x[:, : self.lv_latent_dim] * init_scaling
        # extract encoded std from NN output
        z_std = (
            self.lv_init_std
            - F.softplus(x[:, self.lv_latent_dim :]) * init_scaling  # noqa: E203
        )

        self.z_mu = z_mu
        self.z_log_sigma = torch.log(torch.expm1(z_std))
        weights = z_mu + z_std * weights_eps

        self.log_f_hat_z = calc_log_f_hat(
            weights, z_mu, z_std, prior_variance=self.lv_prior_std**2
        )
        self.log_normalizer_z = calc_log_normalizer(z_mu, z_std)

        return weights

    def calc_log_f_hat_z(self, z, m_z, std_z) -> Tensor:
        """Calculate log_f_hat_z.

        Args:
            z: sampled latent vector [batch_size, lv_latent_dim]
            m_z: mean latent weight [num_training_points, lv_latent_dim]
            std_z: std latent weight [num_training_poinst, lv_latent_dim]

        Returns:
            log_f_hat_z
        """
        v_prior_z = self.lv_prior_std**2
        v_z = std_z**2

        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        return (
            ((v_z - v_prior_z) / (2 * v_prior_z * v_z))[None, ...] * (z**2)
            + (m_z / v_z)[None, ...] * z
        ).sum(2)
