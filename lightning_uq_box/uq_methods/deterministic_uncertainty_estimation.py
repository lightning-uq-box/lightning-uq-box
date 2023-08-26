"""Deterministic Uncertainty Estimation."""
import os
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from torch.utils.data import DataLoader
from torchgeo.trainers.utils import _get_input_layer_name_and_module

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
        input_size: int = None,
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
            input_size: image input size to feature_extractor
            coeff: soft normalization only when sigma larger than coeff should be (0, 1)
        """
        self.input_size = input_size

        self.input_dimensions = self._collect_input_sizes(feature_extractor)
        # spectral normalize the feature extractor layers
        feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations, self.input_dimensions, coeff
        )
        super().__init__(
            feature_extractor,
            gp_layer,
            elbo_fn,
            n_inducing_points,
            optimizer,
            lr_scheduler,
            save_dir,
            quantiles,
        )

    def _collect_input_sizes(self, feature_extractor):
        """Spectral Normalization needs input sizes to each layer."""
        try:
            _, module = _get_input_layer_name_and_module(feature_extractor)
        except UnboundLocalError:
            input_dimensions = {}
            return input_dimensions

        if isinstance(module, torch.nn.Linear):
            input_tensor = torch.zeros(1, module.in_features)
        elif isinstance(module, torch.nn.Conv2d):
            input_tensor = torch.zeros(
                1, module.in_channels, self.input_size, self.input_size
            )

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
