import math

import torch
import torch.nn as nn


def energy_function(y, y_pred, loss_terms, N, alpha=1.0):
    # y=[N,d], y_pred=[N,s,d]

    log_f_hat = loss_terms["log_f_hat"]
    log_Z_prior = loss_terms["log_Z_prior"]
    log_normalizer = loss_terms["log_normalizer"]
    log_normalizer_z = loss_terms["log_normalizer_z"]
    log_f_hat_z = loss_terms["log_f_hat_z"]

    S = y_pred.batch_shape[0]
    n_samples = y_pred.batch_shape[1]
    alpha = torch.tensor(alpha).to(y.device)
    NoverS = torch.tensor(N / S).to(y.device)
    one_over_n_samples = torch.tensor(1 / n_samples).to(y.device)
    return (
        -(1 / alpha)
        * NoverS
        * torch.sum(
            torch.logsumexp(
                alpha
                * (
                    y_pred.log_prob(y[:, None, :]).sum(-1)
                    - (1 / N) * log_f_hat
                    - log_f_hat_z
                ),
                1,
            )
            + torch.log(one_over_n_samples)
        )
        - log_normalizer
        - NoverS * log_normalizer_z
        + log_Z_prior
    )
