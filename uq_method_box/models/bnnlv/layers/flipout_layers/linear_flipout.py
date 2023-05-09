"""Linear Flipout Layer adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from torch import Tensor

__all__ = ["LinearFlipout"]


class LinearFlipout(LinearFlipout):
    """Linear Flipout Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ) -> None:
        """Initialize a new Linear Flipout layer.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            prior_mean: mean of the prior arbitrary distribution to be used on the
                complexity cost
            prior_variance: variance of the prior arbitrary distribution to be used
                on the complexity cost
            posterior_mu_init: init trainable mu parameter representing mean of the
                approximate posterior
            posterior_rho_init: init trainable rho parameter representing the sigma
                of the approximate posterior through softplus function
            bias: if set to False, the layer will not learn an additive bias.
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

    def forward(self, x, return_logs=False):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs+perturbed of layer.
        """
        # gotta double check if we need this next line
        # actually we want to use dnn_to_bnn_some extended for lvs
        if self.dnn_to_bnn_flag:
            return_logs = False

        # sampling delta_W and delta_b
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        delta_weight = sigma_weight * self.eps_weight.data.normal_()

        bias = None
        # get log_normalizer and log_f_hat for biases
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            delta_bias = sigma_bias * self.eps_bias.data.normal_()
            bias = self.mu_bias + delta_bias

        # linear outputs
        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = F.linear(x * sign_input, delta_weight, bias) * sign_output

        # returning outputs + perturbations
        if return_logs:
            return outputs + perturbed_outputs
        return outputs + perturbed_outputs
