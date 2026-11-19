import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


def energy_function(
    y: Tensor, y_pred: Normal, loss_terms: dict[str, Tensor], N: int, alpha: float = 1.0
) -> Tensor:
    """Compute the energy function loss.

    Args:
        y: target of shape [batch_size, output_dim]
        y_pred: BNN model output with shape [batch_size, num_samples, output_dim]
        loss_terms: collected loss terms over the variational layer weights
        N: number of datapoints in dataset
        alpha: alpha divergence value

    Returns:
        energy function loss
    """
    log_f_hat = loss_terms["log_f_hat"]  # ["num_samples"]
    log_Z_prior = loss_terms["log_Z_prior"]  # 0 shape
    log_normalizer = loss_terms["log_normalizer"]  # 0 shape
    log_normalizer_z = loss_terms["log_normalizer_z"]  # 0 shape
    log_f_hat_z = loss_terms["log_f_hat_z"]  # ["S", "num_samples"] shape
  

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
