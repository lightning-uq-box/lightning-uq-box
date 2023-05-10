"""LSTM Variational Layer adapted for Alpha Divergence."""

import math

import torch
from bayesian_torch.layers.variational_layers.rnn_variational import *
from torch import Tensor

from .variational_layers.linear_variational import LinearReparameterization


class LSTMReparameterization(LSTMReparameterization):
    """LSTM Variational Layer adapted for Alpha Divergence."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ):
        """
        Implement LSTM layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers.rnn_variational,
        LSTMReparameterization.

        Parameters:
            prior_mean: mean of the prior arbitrary
                distribution to be used on the complexity cost,
            prior_variance: variance of the prior
                arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: init std for the trainable mu parameter,
                sampled from N(0, posterior_mu_init),
            posterior_rho_init: init std for the trainable rho parameter,
                sampled from N(0, posterior_rho_init),
            in_features: size of each input sample,
            out_features: size of each output sample,
            bias: if set to False, the layer will not learn an additive bias.
                Default: True.
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

        self.ih = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=in_features,
            out_features=out_features * 4,
            bias=bias,
        )

        self.hh = LinearReparameterization(
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            in_features=out_features,
            out_features=out_features * 4,
            bias=bias,
        )

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = (
            self.hh.mu_weight.numel()
            + self.hh.mu_bias.numel()
            + self.ih.mu_bias.numel()
            + self.ih.mu_bias.numel()
        )
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def log_f_hat(self):
        """Compute log_f_hat for energy functional.

        Args:
            self.
        Returns:
            log_f_hat.
        """
        log_f_hat_ih = self.ih.log_f_hat()
        log_f_hat_hh = self.hh.log_f_hat()
        return log_f_hat_ih + log_f_hat_hh

    def log_normalizer(self):
        """Compute log terms for energy functional.

        Args:
            self.
        Returns:
            log_normalizer.
        """
        log_normalizer_ih = self.ih.log_normalizer()
        log_normalizer_hh = self.hh.log_normalizer()

        return log_normalizer_ih + log_normalizer_hh

    def forward(self, X, hidden_states=None):
        """Forward pass through layer.

        Args: self: layer.
            x: input.
        Returns:
            outputs of layer.
        """
        batch_size, seq_size, _ = X.size()

        hidden_seq = []
        c_ts = []

        if hidden_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.out_features).to(X.device),
                torch.zeros(batch_size, self.out_features).to(X.device),
            )
        else:
            h_t, c_t = hidden_states

        HS = self.out_features
        for t in range(seq_size):
            x_t = X[:, t, :]

            ff_i = self.ih(x_t)
            # like a LinearReparameterization layer
            ff_h = self.hh(h_t)
            # like a LinearReparameterization layer
            gates = ff_i + ff_h

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            c_ts.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        c_ts = torch.cat(c_ts, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_ts = c_ts.transpose(0, 1).contiguous()

        return hidden_seq, (hidden_seq, c_ts)
