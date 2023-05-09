"""Linear Variational Layers adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers.linear_variational import *
from torch import Tensor


class LinearReparameterization(LinearReparameterization):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean=0,
        prior_variance=1,
        posterior_mu_init=0,
        posterior_rho_init=-3.0,
        bias=True,
    ):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers.linear_variational,
        LinearReparameterization.

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__(
            in_features,
            out_features,
            prior_mean,
            prior_variance,
            posterior_mu_init,
            posterior_rho_init,
            bias,
        )

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_weight.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single summand in equation 3.16.

        Args:
            w: weight matrix [num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log f hat summed over the parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum()

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute single left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()

    def log_normalizer(self, return_logs=True):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_normalizer.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        # get log_normalizer and log_f_hat for weights
        if return_logs:
            log_normalizer = self.calc_log_normalizer(
                m_W=self.mu_weight, std_W=sigma_weight
            )

        # get log_normalizer for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            if return_logs:
                log_normalizer = log_normalizer + self.calc_log_normalizer(
                    m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_normalizer

    def log_f_hat(self, return_logs=True):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = sigma_weight * self.eps_weight.data.normal_()

        # sampling weight and bias
        weight = self.mu_weight + delta_weight

        # get log_f_hat for weights
        if return_logs:
            log_f_hat = self.calc_log_f_hat(
                w=weight, m_W=self.mu_weight, std_W=sigma_weight
            )

        bias = None
        # get log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias
            if return_logs:
                log_f_hat = log_f_hat + self.calc_log_f_hat(
                    w=bias, m_W=self.mu_bias, std_W=sigma_bias
                )

        return log_f_hat

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + (sigma_weight * self.eps_weight.data.normal_())

        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())

        out = F.linear(input, weight, bias)

        return out
