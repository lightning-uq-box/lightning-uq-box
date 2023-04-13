"""Deterministic Uncertainty Estimation."""

from typing import Any, Dict, List

import torch.nn as nn
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP

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
        gp_args: Dict[str, Any],
        elbo_fn: type[_ApproximateMarginalLogLikelihood],
        n_train_points: int,
        lr: float,
        save_dir: str,
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
        """
        super().__init__(
            feature_extractor, gp_layer, gp_args, elbo_fn, n_train_points, lr, save_dir
        )

        # spectral normalize the feature extractor layers
        self.feature_extractor = spectral_normalize_model_layers(feature_extractor)
