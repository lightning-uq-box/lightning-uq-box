# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Utilities for computing Uncertainties."""

from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from torch import Tensor

# TODO:
# write function to compute nll for GMM for BNNs without moment matching
# e.g. get preds shape [batch_size, n_outputs, num_mc_samples]
# n_outputs are already transformed to be sigma's not log sigmas
# compute sum_(over batch elements x,y) log ( sum_(over samples i)
# (1/sqrt(2 *\pi* sigma(x)_i^2))*exp(-(mu_i(x)-y)^2/(2 sigma(x)_i^2)) )
# def compute_test_nll_on_gmm(std: np.ndarray) -> np.ndarray:
#     """Compute test negative log-likelihood of GMM.

#     Args:
#       mean: prediction of the mean from network,
#         shape [num_mc_samples, batch_size, 1]
#       sigma^2: preds.
#     """
#     return NotImplementedError


def compute_epistemic_uncertainty(sample_mean_preds: Tensor) -> Tensor:
    """Compute epistemic uncertainty as defined in Kendall et al. 2017.

    Equation (9) left hand side. Gaussian Mixture Model assumption.

    Args:
      sample_mean_preds: sample mean predictions N x num_samples

    Returns:
      epistemic uncertainty for each sample
    """
    right_term = sample_mean_preds.mean(1) ** 2
    return torch.sqrt((sample_mean_preds**2).mean(axis=1) - right_term)


def compute_aleatoric_uncertainty(sample_sigma_preds: Tensor) -> Tensor:
    """Compute aleatoric uncertainty as defined in Kendall et al. 2017.

    Equation (9) right hand side. Gaussian Mixture Model assumption.

    Args:
      sample_sigma_preds: sample sigma predictions N x num_samples

    Returns:
      aleatoric uncertainty for each sample
    """
    return torch.sqrt(sample_sigma_preds.mean(-1))


def compute_predictive_uncertainty(
    sample_mean_preds: Tensor, sample_sigma_preds: Tensor
) -> Tensor:
    """Compute predictive uncertainty as defined in Kendall et al. 2017.

    Equation (9). Gaussian Mixture Model.

    Args:
      sample_mean_preds: sample mean predictions N x num_samples
      sample_sigma_preds: sample sigma predictions N x num_samples

    Returns:
      predictive uncertainty for each sample
    """
    return torch.sqrt(
        sample_sigma_preds.mean(-1)
        + (sample_mean_preds**2).mean(-1)
        - (sample_mean_preds.mean(-1) ** 2)
    )


def compute_sample_mean_std_from_quantile(
    inter_range_quantiles: np.ndarray, quantiles: list[float]
) -> np.ndarray:
    """Compute sample mean and std from inter quantiles.

    Taken from: https://stats.stackexchange.com/questions/256456/
    how-to-calculate-mean-and-standard-deviation-from-median-and-quartiles,
    https://stats.stackexchange.com/questions/240364/how-to-obtain-the-mean-
    for-a-normal-distribution-given-its-quartiles

    Args:
      inter_range_quantiles: N x num_quantiles
      quantiles: specifying the corresponding quantiles

    Returns:
      tuple of estimated mean and std for each sample
    """
    # n = inter_range_quantiles.shape[0]
    mu = inter_range_quantiles.mean(-1)
    upper_q = max(quantiles)
    # lower_q = min(quantiles)

    std = (inter_range_quantiles[:, -1] - inter_range_quantiles[:, 0]) / (
        2 * stats.norm.ppf(upper_q)
    )
    return mu, std


def compute_quantiles_from_std(
    means: np.array, stds: np.array, quantiles: list[float]
) -> np.ndarray:
    """Compute quantiles from standard deviations assuming a Gaussian.

    Args:
      means: mean prediction vector
      stds: mean std vector
      quantiles: desired quantiles

    Returns:
      desired quantiles stacked
    """
    # define the normal distribution and PDF
    dists = [stats.norm(loc=mean, scale=sigma) for mean, sigma in np.c_[means, stds]]

    # calculate PPFs
    ppfs = defaultdict(list)
    for dist in dists:
        for ppf in quantiles:
            p = dist.ppf(ppf)
            ppfs[ppf].append(p)

    quantiles = np.stack(list(ppfs.values()), axis=-1)
    return quantiles


def compute_empirical_coverage(quantile_preds: np.ndarray, targets: np.ndarray):
    """Compute the empirical coverage.

    Args:
      quantile_preds: predicted quantiles
      labels: regression targets

    Returns:
      computed empirical coverage over all samples
    """
    targets = targets.squeeze()
    return (
        (targets >= quantile_preds[:, 0]) & (targets <= quantile_preds[:, -1])
    ).mean()
