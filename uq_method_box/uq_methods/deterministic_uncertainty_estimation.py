"""Deterministic Uncertainty Estimation."""

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
        train_loader: DataLoader,
        n_inducing_points: int,
        optimizer: type[torch.optim.Optimizer],
        n_power_iterations: int = 10,
        save_dir: str = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Deterministic Uncertainty Estimation Model.

        Args:
            feature_extractor:
            gp_layer: gpytorch module that takes extracted features as inputs
            gp_args: arguments to initializ the gp_layer
            elbo_fn: gpytorch elbo functions
            n_train_points: number of training points necessary f
                or Gpytorch elbo function
        """
        super().__init__(
            feature_extractor,
            gp_layer,
            elbo_fn,
            train_loader,
            n_inducing_points,
            optimizer,
            save_dir,
        )

        # spectral normalize the feature extractor layers
        self.feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations
        )
