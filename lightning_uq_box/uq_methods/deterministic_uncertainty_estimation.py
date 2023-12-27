# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Deterministic Uncertainty Estimation (DUE)."""

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from .deep_kernel_learning import DKLClassification, DKLRegression
from .spectral_normalized_layers import (
    collect_input_sizes,
    spectral_normalize_model_layers,
)


class DUERegression(DKLRegression):
    """Deterministic Uncertainty Estimation (DUE) for Regression.

    If you use this model in your research, please cite the following paper:x

    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        n_inducing_points: int,
        input_size: int,
        num_targets: int = 1,
        gp_kernel: str = "RBF",
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deterministic Uncertainty Estimation Model.

        Initialize a new Deep Kernel Learning Model for Regression.

        Args:
            feature_extractor: feature extractor model
            n_inducing_points: number of inducing points
            num_targets: number of targets
            gp_kernel: GP kernel choice, supports one of
                'RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']
            input_size: image input size of data to the model
            coeff: soft normalization only when sigma larger than coeff,
                should be (0, 1)
            n_power_iterations: number of power iterations for spectral normalization
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.input_size = input_size

        self.input_dimensions = collect_input_sizes(feature_extractor, self.input_size)
        # spectral normalize the feature extractor layers
        feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations, self.input_dimensions, coeff
        )

        super().__init__(
            feature_extractor,
            n_inducing_points,
            num_targets,
            gp_kernel,
            optimizer,
            lr_scheduler,
        )


class DUEClassification(DKLClassification):
    """Deterministic Uncertainty Estimation (DUE) Model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        n_inducing_points: int,
        input_size: int,
        num_classes: int,
        gp_kernel: str = "RBF",
        task: str = "multiclass",
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deterministic Uncertainty Estimation Model.

        Args:
            feature_extractor: feature extractor model
            n_inducing_points: number of inducing points
            input_size: image input size of data to the model
            num_classes: number of classes
            gp_kernel: GP kernel choice, supports one of
                'RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            coeff: soft normalization only when sigma larger than coeff should be (0, 1)
            n_power_iterations: number of power iterations for spectral normalization
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.input_size = input_size

        self.input_dimensions = collect_input_sizes(feature_extractor, self.input_size)
        # spectral normalize the feature extractor layers
        feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations, self.input_dimensions, coeff
        )

        super().__init__(
            feature_extractor,
            n_inducing_points,
            num_classes,
            task,
            gp_kernel,
            optimizer,
            lr_scheduler,
        )
