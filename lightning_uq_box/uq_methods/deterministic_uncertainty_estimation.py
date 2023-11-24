"""Deterministic Uncertainty Estimation."""

from typing import Dict

import torch
import torch.nn as nn
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from .deep_kernel_learning import DKLClassification, DKLRegression
from .spectral_normalized_layers import spectral_normalize_model_layers
from .utils import _get_input_layer_name_and_module


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
            gp_kernel: GP kernel, one of ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ]
            inputs_size: reature input size of data to the model
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
            gp_layer: Gaussian Process layer
            elbo_fn: gpytorch elbo function used for optimization
            n_inducing_points: number of inducing points
            optimizer: optimizer used for training
            inputs_size: input size of image data to the model
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            coeff: soft normalization only when sigma larger than coeff should be (0, 1)
            n_power_iterations: number of power iterations for spectral normalization
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


def collect_input_sizes(feature_extractor, input_size) -> Dict[str, torch.Size]:
    """Spectral Normalization needs input sizes to each layer.

    Args:
        feature_extractor: feature extractor model
        input_size: input size of image data to the model

    Returns:
        input_dimensions: dictionary of input dimensions to each layer
    """
    _, module = _get_input_layer_name_and_module(feature_extractor)

    if isinstance(module, torch.nn.Linear):
        input_tensor = torch.zeros(1, module.in_features)
    elif isinstance(module, torch.nn.Conv2d):
        input_tensor = torch.zeros(1, module.in_channels, input_size, input_size)

    input_dimensions = {}

    hook_handles = []

    def hook_fn(layer_name):
        def forward_hook(module, input, output):
            layer_name = f"{id(module)}"  # register unique id for each module
            input_dimensions[layer_name] = input[0].shape[
                1:
            ]  # Assuming input is a tuple

            input_dimensions[layer_name] = input[0].shape[
                1:
            ]  # Assuming input is a tuple

        return forward_hook

    # Register the forward hooks for each convolutional layer in the model
    for name, module in feature_extractor.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = hook_fn(name)
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)

    # Perform a forward pass
    _ = feature_extractor(input_tensor)

    # Remove the hooks
    for handle in hook_handles:
        handle.remove()

    return input_dimensions
