"""Latent Variable Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from uq_method_box.models.bnn_layers.utils import calc_log_normalizer


def calc_log_f_hat_z(
    z: Tensor, m_z: Tensor, std_z: Tensor, prior_sigma: float
) -> Tensor:
    """Compute single summand in equation 3.16.

    Args:
        z: samples from lv [mb_size, lv_latent_dim]
        m_z: mean of latent variables [mb_size, lv_latent_dim]
        std_z: sigma weight matrix of latent variables [mb_size, lv_latent_dim]
        prior_sigma: Prior variance of latent variables

    Returns:
        log f hat summed over the latent dimension
    """
    v_z = std_z**2
    m_z = m_z
    # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
    # \lambda is (\lambda_q - \lambda_prior) / N
    # assuming prior mean is 0 and moving N calculation outside

    return (
        ((v_z - prior_sigma) / (2 * prior_sigma * v_z)) * (z**2) + (m_z / v_z) * z
    ).sum(-1)


class LatentVariableNetwork(nn.Module):
    """Latent Variable Network for BNN+LV."""

    def __init__(
        self,
        net: nn.Module,
        num_training_points: int,
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 5.0,
        lv_latent_dim: int = 1,
        lv_init_std: float = 1.0,
        init_scaling: float = 0.01,
    ) -> None:
        """Initialize a new Latent Variable Network.

        Used for amortized inference, as in eq. (3.22)
        in [1].
        [1]: Depeweg, Stefan.
        Modeling epistemic and aleatoric uncertainty
        with Bayesian neural networks and latent variables.
        Diss. Technische Universität München, 2019.

        Args:
            net: nn.Module, network that is deterministic,
                i.e. the latent variable net.
            num_training_points:
            lv_prior_mu: Prior mean for latent variables,
                default: 0.0.
            lv_prior_std: Prior standard deviation for latent variables,
                default: sqrt(d), where d is the dimension of the
                features in the net layer to which the lv's
                are added.
            lv_init_std: Initialized standard deviation of lv's.
            lv_latent_dim: dimension of latent variables z.
                Default: 1.
            init_scaling: factor to make sure q(z)
                is close to N(0,1) at initialization
        """
        super().__init__()

        self.net = net
        self.num_training_points = num_training_points
        self.lv_prior_mu = lv_prior_mu
        self.lv_prior_std = lv_prior_std

        self.lv_init_std = lv_init_std
        self.lv_latent_dim = lv_latent_dim

        self.init_scaling = init_scaling

        self.z_mu = torch.tensor(0.0)

        self.log_f_hat_z = None
        self.log_normalizer_z = None
        self.weight_eps = None

    def fix_randomness(self) -> None:
        """Fix the randomness of reparameterization trick."""
        # weight eps vector large enough to cover entire
        # dataset for full batch theoretically
        self.weight_eps = torch.randn(self.num_training_points, self.lv_latent_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass where latent vector is sampled.

        Args:
            x: input X [batch_size, num_features]
            y: target y [batch_size, target_dim]

        Returns:
            z vector of dimension [batch_size, latent_dim]
        """
        if self.weight_eps is None:
            if x.dim() == 3:
                weights_eps = torch.randn(
                    x.shape[0], x.shape[1], self.lv_latent_dim
                ).to(x.device)
            else:
                weights_eps = torch.randn(x.shape[0], self.lv_latent_dim).to(x.device)
        else:
            # always extract the same weight_eps since we
            # want it fixed
            weights_eps = self.weight_eps[: x.shape[0], :].to(x.device)

        
        if x.dim() == 3:
            xy = torch.cat([x[0], y[0]], dim=-1)
        else:
            xy = torch.cat([x, y], dim=-1)

        # pass through NN
        # so here we are passing the input through the whole lv inf net
        x = self.net(xy)  # out [batch_size, lv_latent_dim*2]


        # make sure q(z) is close to N(0,1) at initialization
        # extract encoded mean from NN output
        z_mu = x[..., : self.lv_latent_dim] * self.init_scaling
        # extract encoded std from NN output
        # shouldn't z_std be pos, why dont we use unconstrained
        # optimatization wit z_rho and then
        # use z_std = torch.log1p(torch.exp(self.z_rho))?

        eps = 1e-6
        init_shift = 3.
        z_std = eps + self.lv_prior_std* (1 - F.sigmoid(x[..., self.lv_latent_dim:]-init_shift)) 
        # these are lv network outputs,
        # as in eq. (3.22), [1]
        latent_samples = z_mu + z_std * weights_eps

        self.log_f_hat_z = calc_log_f_hat_z(
            latent_samples, z_mu, z_std, prior_sigma=self.lv_prior_std**2
        )
        self.log_normalizer_z = calc_log_normalizer(z_mu, z_std)

        return latent_samples
