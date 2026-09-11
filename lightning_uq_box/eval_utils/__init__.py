# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Evaluation Utils for UQ-Regression-Box."""

from .joint_prediction import (
    average_sampled_log_likelihood,
    categorical_kl,
    dyadic_batch_indices,
    joint_log_likelihood,
    joint_log_loss_dyadic,
    marginal_log_likelihood,
    marginal_log_loss,
)
from .uq_computation import (
    compute_aleatoric_uncertainty,
    compute_empirical_coverage,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
    compute_sample_mean_std_from_quantile,
)

__all__ = (
    # evaluation utils 1d regression
    "compute_aleatoric_uncertainty",
    "compute_empirical_coverage",
    "compute_epistemic_uncertainty",
    "compute_predictive_uncertainty",
    "compute_quantiles_from_std",
    "compute_sample_mean_std_from_quantile",
    # joint prediction metrics
    "average_sampled_log_likelihood",
    "joint_log_likelihood",
    "marginal_log_likelihood",
    "dyadic_batch_indices",
    "joint_log_loss_dyadic",
    "marginal_log_loss",
    "categorical_kl",
)
