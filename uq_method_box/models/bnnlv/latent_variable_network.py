"""Latent Variable Network."""


import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

    def fix_randomness(self, n_samples=Optional[int]) -> None:
        """Fix the randomness of reparameterization trick.

        Args:
            n_samples: number of samples to draw
        """
        if n_samples is None:
            n_samples = self.n_samples
        self.weight_eps = torch.randn_like(
            n_samples, self.num_training_points, self.lv_latent_dim
        ).to(self.device)

    def forward(self, x: Tensor, latent_idx: Tensor, n_samples=Optional[int]) -> Tensor:
        """Forward pass where latent vector is sampled.

        Args:
            x: input X [batch_size, num_features]
            latent_idx: latent vector [batch_size, 1] containing the latent indices
            n_samples: number of samples to draw

        Returns:
            z vector of dimension [batch_size, latent_dim]
        """
        if n_samples is None:
            n_samples = self.n_samples
        if self.weight_eps is None:
            weights_eps = torch.randn(n_samples, x.shape[0], self.lv_latent_dim).to(
                x.device
            )
        else:
            weights_eps = self.weight_eps[:, : x.shape[0], :]

        xy = torch.cat([x, latent_idx], dim=-1)
        # pass through network
        x = self.net(xy)

        # make sure q(z) is close to N(0,1) at initialization
        init_scaling = 0.1
        z_mu = latent_idx * init_scaling
        z_std = self.lv_init_std - F.softplus(latent_idx) * init_scaling

        self.z_mu = z_mu
        self.z_log_sigma = torch.log(torch.expm1(z_std))
        weights = z_mu + z_std * weights_eps

        self.log_f_hat_z = self.calc_log_f_hat_z(weights, z_mu, z_std).transpose(0, 1)
        self.log_normalizer_z = self.calc_log_normalizer_q_z(z_mu, z_std)

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

    def calc_log_normalizer_q_z(self, m_z: Tensor, std_z: Tensor) -> Tensor:
        """Calculate log normalizer approximate latent variable dist.

        Args:
            m_z: mean latent weight [num_training_points, lv_latent_dim]
            std_z: std latent weight [num_training_poinst, lv_latent_dim]

        Returns:
            log_normalizer_q_z
        """
        v_z = std_z**2
        return (0.5 * torch.log(v_z * 2 * math.pi) + 0.5 * m_z**2 / v_z).sum()
