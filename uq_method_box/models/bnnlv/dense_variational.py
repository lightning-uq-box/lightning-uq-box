import math
import numpy as np
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.distributions as dist


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
        self.weight_eps = torch.randn_like(n_samples, self.N, self.latent_dim).to(
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


class DenseVariational(nn.Module):
    DEFAULT_CONFIG = {
        "layer_prior_mu": 0.0,
        "layer_prior_std": 1.0,
        "layer_init_std": 0.1,
        "layer_init_mu_std": 0.1,
        "n_samples": 25,
        "device": None,
    }

    def __init__(self, in_features, out_features, config={}):
        super(DenseVariational, self).__init__()
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.in_features = in_features
        self.out_features = out_features
        self.device = self.config["device"]
        self.log_var_init = np.log(np.expm1(self.config["layer_init_std"]))

        self.weight_mu = nn.Parameter(
            nn.init.normal_(
                torch.empty((out_features, in_features)),
                0,
                self.config["layer_init_mu_std"],
            )
        )
        # std = Softplus(rho) = log(1 + exp(rho))
        self.weight_log_sigma = nn.Parameter(
            nn.init.constant_(
                torch.empty((out_features, in_features)), self.log_var_init
            )
        )

        # initialize mu and rho parameters for the layer's bias
        self.bias_mu = nn.Parameter(
            nn.init.normal_(
                torch.empty((out_features, 1)), 0.0, self.config["layer_init_mu_std"]
            )
        )
        self.bias_log_sigma = nn.Parameter(
            nn.init.constant_(torch.empty((out_features, 1)), self.log_var_init)
        )

        self.log_Z_prior = self.calc_log_Z_prior()
        self.log_f_hat = None
        self.log_normalizer = None
        self.weight_eps = None
        self.bias_eps = None

    def fix_randomness(self, n_samples=None):
        if n_samples is None:
            n_samples = self.config["n_samples"]
        self.weight_eps = torch.randn(
            n_samples, self.out_features, self.in_features
        ).to(self.device)
        self.bias_eps = torch.randn(n_samples, self.out_features, 1).to(self.device)

    def unfix_randomness(self):
        self.weight_eps = None
        self.bias_eps = None

    def sample_weights(self, n_samples):
        if self.weight_eps is None:
            weights_eps = torch.randn(
                n_samples, self.out_features, self.in_features
            ).to(self.device)
            bias_eps = torch.randn(n_samples, self.out_features, 1).to(self.device)

        else:
            weights_eps = self.weight_eps[:n_samples]
            bias_eps = self.bias_eps[:n_samples]

        weight_sigma = F.softplus(self.weight_log_sigma)
        bias_sigma = F.softplus(self.bias_log_sigma)
        weight_sample = self.weight_mu + weights_eps * weight_sigma
        bias_sample = self.bias_mu + bias_eps * bias_sigma

        all_params_mu = torch.cat([self.weight_mu.flatten(), self.bias_mu.flatten()])
        all_params_sigma = torch.cat([weight_sigma.flatten(), bias_sigma.flatten()])
        all_sample = torch.cat([weight_sample.flatten(1), bias_sample.flatten(1)], 1)

        self.log_normalizer = self.calc_log_normalizer(all_params_mu, all_params_sigma)
        self.log_f_hat = self.calc_log_f_hat(
            all_sample, all_params_mu, all_params_sigma
        )
        return weight_sample, bias_sample

    def forward(self, input):
        input = input.to(self.device)

        inp_shape = input.shape
        if len(inp_shape) == 2:
            n_samples = self.config["n_samples"]
            input = input.unsqueeze(0).repeat(n_samples, 1, 1)
        else:
            n_samples = inp_shape[0]

        weight, bias = self.sample_weights(n_samples)

        state = input.matmul(weight.transpose(-1, -2)) + bias.transpose(
            1, 2
        )  # n_samples x batch_size x out_features

        return state

    def calc_log_Z_prior(self):
        n_params = self.weight_mu.numel() + self.bias_mu.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.config["layer_prior_std"] ** 2 * 2 * math.pi)
        ).to(self.device)

    def calc_log_f_hat(self, w, m_W, std_W):
        v_prior = self.config["layer_prior_std"] ** 2

        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - v_prior) / (2 * v_prior * v_W)) * (w**2) + (m_W / v_W) * w
        ).sum(axis=1)

    def calc_log_normalizer(self, m_W, std_W):
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
