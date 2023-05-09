"""LSTM Variational Layers adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.variational_layers.rnn_variational import *
from torch import Tensor


class LSTMReparameterization(LSTMReparameterization):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias=True,
    ):
        """
        Implements LSTM layer with reparameterization trick.

        Inherits from bayesian_torch.layers.variational_layers.rnn_variational,
        LSTMReparameterization.

        Parameters:
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init std for the trainable mu parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init: float -> init std for the trainable rho parameter, sampled from N(0, posterior_rho_init),
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
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


def forward(self, X, hidden_states=None, return_kl=True):
    if self.dnn_to_bnn_flag:
        return_kl = False

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
    kl = 0
    for t in range(seq_size):
        x_t = X[:, t, :]

        ff_i, kl_i = self.ih(x_t)
        ff_h, kl_h = self.hh(h_t)
        gates = ff_i + ff_h

        kl += kl_i + kl_h

        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS : HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2 : HS * 3]),
            torch.sigmoid(gates[:, HS * 3 :]),  # output
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
