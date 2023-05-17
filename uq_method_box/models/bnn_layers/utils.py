"""Utility functions for BNN Layers."""

import math

import torch
from torch import Tensor


def calc_log_f_hat(
    w: Tensor, m_W: Tensor, std_W: Tensor, prior_variance: float
) -> Tensor:
    """Compute single summand in equation 3.16.

    Args:
        w: weight matrix [num_params]
        m_W: mean weight matrix at current iteration [num_params]
        std_W: sigma weight matrix at current iteration [num_params]
        prior_variance: Prior variance of weights

    Returns:
        log f hat summed over the parameters shape 0
    """
    v_W = std_W**2
    m_W = m_W
    # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
    # \lambda is (\lambda_q - \lambda_prior) / N
    # assuming prior mean is 0 and moving N calculation outside
    return (
        ((v_W - prior_variance) / (2 * prior_variance * v_W)) * (w**2)
        + (m_W / v_W) * w
    ).sum()


def calc_log_normalizer(m_W: Tensor, std_W: Tensor) -> Tensor:
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
