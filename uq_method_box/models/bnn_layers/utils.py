"""Utility functions for BNN Layers."""

import math

import torch
from torch import Tensor


def calc_log_f_hat(w: Tensor, m_W: Tensor, std_W: Tensor, prior_sigma: float) -> Tensor:
    """Compute single summand in equation 3.16.

    Args:
        w: weight matrix [num_params]
        m_W: mean weight matrix at current iteration [num_params]
        std_W: sigma weight matrix at current iteration [num_params]
        prior_sigma: Prior variance of weights

    Returns:
        log f hat summed over the parameters shape 0
    """
    v_W = std_W**2
    # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
    # \lambda is (\lambda_q - \lambda_prior) / N
    # assuming prior mean is 0 and moving N calculation outside
    out =  (
        ((v_W - prior_sigma) / (2 * prior_sigma * v_W)) * (w**2) + (m_W / v_W) * w
    )
    # sum out over all dimension except first

    return  torch.sum(out, dim=tuple(range(1, out.ndim)))
 


def calc_log_normalizer(m_W: Tensor, std_W: Tensor) -> Tensor:
    """Compute single left summand of 3.18.

    Args:
        m_W: mean weight matrix at current iteration [num_params]
        std_W: sigma weight matrix at current iteration [num_params]

    Returns:
        log normalizer summed over all layer parameters shape 0
    """
    v_W = std_W**2
    return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
