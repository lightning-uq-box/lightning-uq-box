import math

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear

class LV(nn.Module):

    DEFAULT_CONFIG = {
        "lv_prior_mu": 0.0,
        "lv_prior_std": 1.0,
        "lv_init_mu": 0.0,
        "lv_init_std": 1.0,
        "lv_latent_dim": 1,
        "n_samples": 25,
        "device": None,
    }

    def __init__(self, N, config={}):
        super(LV, self).__init__()
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.device = self.config["device"]
        self.N = N
        self.log_var_init = np.log(np.expm1(self.config["lv_init_std"]))

        self.z_mu = nn.Parameter(
            nn.init.constant_(
                torch.empty((N, self.config["lv_latent_dim"])),
                self.config["lv_init_mu"],
            )
        )
        self.z_log_sigma = nn.Parameter(
            nn.init.constant_(
                torch.empty((N, self.config["lv_latent_dim"])), self.log_var_init
            )
        )

        self.log_f_hat_z = None
        self.log_normalizer_z = None
        self.weight_eps = None

    def fix_randomness(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        self.weight_eps = torch.randn_like(n_samples, self.N, self.config['lv_latent_dim']).to(
            self.device
        )

    def unfix_randomness(self):
        self.weight_eps = None

    def sample_weights(self, ind, n_samples=None):
        if n_samples is None:
            n_samples = self.config["n_samples"]
        if self.weight_eps is None:
            weights_eps = torch.randn(
                n_samples, ind.shape[0], self.config["lv_latent_dim"]
            ).to(self.device)
        else:
            weights_eps = self.weight_eps[:, ind, :]
        z_mu = self.z_mu[ind, :]
        z_std = F.softplus(self.z_log_sigma[ind, :])

        weights = z_mu + z_std * weights_eps
        self.log_f_hat_z = self.calc_log_f_hat_z(weights, z_mu, z_std).transpose(
            0, 1
        )  # [N,S]
        self.log_normalizer_z = self.calc_log_normalizer_q_z(z_mu, z_std)
        return weights

    def forward(self, input):
        inp_shape = input.shape
        ind = input.reshape(-1).long()

        z = self.sample_weights(ind)

        return z

    def calc_log_f_hat_z(self, z, m_z, std_z):
        v_prior_z = self.config["lv_prior_std"] ** 2
        v_z = std_z**2

        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        return (
            ((v_z - v_prior_z) / (2 * v_prior_z * v_z))[None, ...] * (z**2)
            + (m_z / v_z)[None, ...] * z
        ).sum(2)

    def calc_log_normalizer_q_z(self, m_z, std_z):
        v_prior_z = self.config["lv_prior_std"] ** 2
        v_z = std_z**2
        return (0.5 * torch.log(v_z * 2 * math.pi) + 0.5 * m_z**2 / v_z).sum()


class LV_inf_net(LV):
    DEFAULT_CONFIG = {
        "lv_prior_mu": 0.0,
        "lv_prior_std": 1.0,
        "lv_init_mu": 0.0,
        "lv_init_std": 1.0,
        "lv_latent_dim": 1,
        "n_samples": 25,
        "device": None,
        "lv_inf_graph": [20, 20],
        
    }
    def __init__(self, in_features, out_features, config={}):
        nn.Module.__init__(self)

        self.config = {**self.DEFAULT_CONFIG, **config}
        self.device = self.config["device"]
        self.in_features = in_features
        self.out_features = out_features
        self.log_var_init = np.log(np.expm1(self.config["lv_init_std"]))


        self.net = nn.ModuleList(
            [Linear(in_features+out_features, self.config["lv_inf_graph"][0], self.config)]
            + [
                Linear(
                    self.config["lv_inf_graph"][i], self.config["lv_inf_graph"][i + 1], self.config
                )
                for i in range(len(self.config["lv_inf_graph"]) - 1)
            ]
            + [Linear(self.config["lv_inf_graph"][-1], self.config['lv_latent_dim']*2, self.config)]
        )
        self.log_f_hat_z = None
        self.log_normalizer_z = None
        self.weight_eps = None
        self.z_mu = torch.tensor(0.0)
        self.z_log_sigma = torch.tensor(np.log(np.expm1(self.config["lv_init_std"])))

    def fix_randomness(self, n_samples=None,N=100):
        if n_samples is None:
            n_samples = self.n_samples
        self.weight_eps = torch.randn_like(n_samples, N, self.config['lv_latent_dim']).to(
            self.device
        )

    def unfix_randomness(self):
        self.weight_eps = None

    def sample_weights(self, xy, n_samples=None):
        if n_samples is None:
            n_samples = self.config["n_samples"]
        if self.weight_eps is None:
            weights_eps = torch.randn(
                n_samples, xy.shape[0], self.config["lv_latent_dim"]
            ).to(self.device)
        else:
            weights_eps = self.weight_eps[:, :xy.shape[0], :]

        x = self.net[0](xy)
        x = F.relu(x)
        for i in range(1, len(self.net)-1):
            x = self.net[i](x)
            x = F.relu(x)
        x = self.net[-1](x)

        
        # make sure q(z) is close to N(0,1) at initialization
        init_scaling = 0.1
        z_mu = x[:, :self.config['lv_latent_dim']] * init_scaling
        z_std = self.config["lv_init_std"] - F.softplus(x[:, self.config['lv_latent_dim']:])*init_scaling
        
        self.z_mu = z_mu
        self.z_log_sigma = torch.log(torch.expm1(z_std))
        weights = z_mu + z_std * weights_eps

        self.log_f_hat_z = self.calc_log_f_hat_z(weights, z_mu, z_std).transpose(
            0, 1
        )  
        self.log_normalizer_z = self.calc_log_normalizer_q_z(z_mu, z_std)
        return weights
    
    def forward(self, input):
        xy = torch.concat(input, axis=1)
        z = self.sample_weights(xy)
        return z