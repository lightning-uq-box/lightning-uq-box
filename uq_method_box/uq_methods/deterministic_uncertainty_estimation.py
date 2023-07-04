"""Deterministic Uncertainty Estimation."""
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from torch.utils.data import DataLoader

from .deep_kernel_learning import DeepKernelLearningModel
from .spectral_normalized_layers import spectral_normalize_model_layers


class DUEModel(DeepKernelLearningModel):
    """Deterministic Uncertainty Estimation (DUE) Model.

    If you use this model in your research, please cite the following paper:x

    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        gp_layer: type[ApproximateGP],
        elbo_fn: type[_ApproximateMarginalLogLikelihood],
        n_inducing_points: int,
        optimizer: type[torch.optim.Optimizer],
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        save_dir: str = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Deterministic Uncertainty Estimation Model.

        Args:
            feature_extractor:
            gp_layer: gpytorch module that takes extracted features as inputs
            gp_args: arguments to initializ the gp_layer
            elbo_fn: gpytorch elbo functions
            n_train_points: number of training points necessary f
                or Gpytorch elbo function
            coeff: soft normalization only when sigma larger than coeff should be (0, 1)
        """
        # spectral normalize the feature extractor layers
        feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations, coeff
        )
        super().__init__(
            feature_extractor,
            gp_layer,
            elbo_fn,
            train_loader,
            optimizer,
            lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
            save_dir,
            quantiles
        )
