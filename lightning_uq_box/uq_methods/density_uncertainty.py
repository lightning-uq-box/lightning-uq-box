# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Density Uncertainty Layer Model."""

import os
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam as Adam

from lightning_uq_box.models.density_layers import DensityConv2d, DensityLinear

from .base import DeterministicModel
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    map_stochastic_modules,
    process_classification_prediction,
    process_regression_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


def get_density_linear_layer(
    params: dict[str, Any], linear_layer: nn.Linear
) -> nn.Module:
    """Convert deterministic linear layer to linear density layer."""
    return DensityLinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        **params,
    )


def get_density_conv_layer(
    params: dict[str, Any], conv_layer: nn.Conv1d | nn.Conv2d | nn.Conv3d
) -> nn.Module:
    """Convert deterministic convolutional layer to convolutional density layer."""
    return DensityConv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
        **params,
    )


def convert_deterministic_to_density(
    deterministic_model: nn.Module,
    density_parameters: dict[str, Any],
    stochastic_module_names: list[str],
) -> None:
    """Replace linear and conv. layers with density layers.

    Args:
        deterministic_model: deterministic pytorch model
        density_parameters: dictionary of density layer parameters
        stochastic_module_names: list of module names that should become density
            layers
    """
    for name in stochastic_module_names:
        layer_names = name.split(".")
        current_module = deterministic_model
        for l_name in layer_names[:-1]:
            current_module = dict(current_module.named_children())[l_name]

        target_layer_name = layer_names[-1]
        current_layer = dict(current_module.named_children())[target_layer_name]

        if "Conv" in current_layer.__class__.__name__:
            setattr(
                current_module,
                target_layer_name,
                get_density_conv_layer(density_parameters, current_layer),
            )
        elif "Linear" in current_layer.__class__.__name__:
            setattr(
                current_module,
                target_layer_name,
                get_density_linear_layer(density_parameters, current_layer),
            )
        else:
            pass


class DensityLayerModelBase(DeterministicModel):
    """Density Layer Model.

    If you use this module in your work, please cite the following paper:

    * https://arxiv.org/abs/2306.12497
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        prior_std: float = 0.1,
        posterior_std_init: float = 1e-3,
        kl_beta: float = 1.0,
        ll_scale: float = 0.01,
        pretrain_epochs: int = 0,
        num_samples_test: int = 1,
        stochastic_module_names: list[int | str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a Density Layer Model.

        Args:
            model: PyTorch model that will be converted to a Density Layer Model.
            loss_fn: Loss function used for the target minimization, could be
                a custom loss function depending on the regression or classification task.
            prior_std: Standard deviation of the prior.
            posterior_std_init: Initial standard deviation of the posterior.
            kl_beta: KL divergence weight.
            ll_scale: Log likelihood scaling factor.
            pretrain_epochs: Number of pretraining epochs for generative energy model,
                which can stabilize training, before switching to normal training regime
                that includes KL divergence.
            num_samples_test: Number of samples to use for test time predictions.
            stochastic_module_names: List of module names that should become density layers.
            freeze_backbone: If True, freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.

        Raises:
            AssertionError: If num_samples_test is less than or equal to 0.
        """
        self.density_layer_args = {
            "prior_std": prior_std,
            "posterior_std_init": posterior_std_init,
        }
        self.stochastic_module_names = map_stochastic_modules(
            model, stochastic_module_names
        )

        self._setup_model(model)

        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

        self.kl_beta = kl_beta
        self.ll_scale = ll_scale
        self.pretrain_epochs = pretrain_epochs
        assert num_samples_test > 0, "num_samples_test must be greater than 0"
        self.num_samples_test = num_samples_test

    def setup_task(self) -> None:
        """Set up task."""
        pass

    def _setup_model(self, model: nn.Module) -> None:
        """Setup the model by converting layers to Density Layers.

        Args:
            model: PyTorch model that will be converted to a Density Layer Model.
        """
        convert_deterministic_to_density(
            model, self.density_layer_args, self.stochastic_module_names
        )

    def compute_kl_divergence(self) -> Tensor:
        """Compute the KL divergence of the model."""
        kl_loss = []
        for layer in self.modules():
            if hasattr(layer, "compute_kl_div"):
                kl_loss.append(layer.compute_kl_div())
        return sum(kl_loss)

    def gather_loglikelihood(self) -> Tensor:
        """Gather loglikelihood terms from the density layers."""
        loglikelihoods = []
        for _, layer in self.named_modules():
            if hasattr(layer, "loglikelihood"):
                loglikelihoods.append(layer.loglikelihood.mean())
        return sum(loglikelihoods)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        y_hat = self.model(X)

        criterion_loss = self.loss_fn(y_hat, y)

        loglikelihood = self.gather_loglikelihood()

        loss = loss = criterion_loss - self.ll_scale * loglikelihood

        if self.current_epoch >= self.pretrain_epochs:
            kl_div = self.compute_kl_divergence()
            # TODO KL multiply factor
            loss += self.kl_beta * kl_div
            self.log("train_kl_div", kl_div * self.kl_beta)

        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        y_hat = self.model(X)

        criterion_loss = self.loss_fn(y_hat, y)

        loglikelihood = self.gather_loglikelihood()

        loss = criterion_loss - self.ll_scale * loglikelihood

        if self.current_epoch >= self.pretrain_epochs:
            kl_div = self.compute_kl_divergence()
            loss += self.kl_beta * kl_div

        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dictionary
        """
        return super().test_step(batch, batch_idx, dataloader_idx)


class DensityLayerModelRegression(DensityLayerModelBase):
    """Density Layer Model for Regression Tasks.

    If you use this module in your work, please cite the following paper:

    * https://arxiv.org/abs/2306.12497
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        prior_std: float = 0.1,
        posterior_std_init: float = 0.001,
        kl_beta: float = 1,
        ll_scale: float = 0.01,
        pretrain_epochs: int = 0,
        num_samples_test: int = 5,
        stochastic_module_names: list[int | str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a Density Layer Model for Regression Tasks.

        Args:
            model: PyTorch model that will be converted to a Density Layer Model.
            loss_fn: Loss function used for the target minimization, defaults to
                MSE Loss.
            prior_std: Standard deviation of the prior.
            posterior_std_init: Initial standard deviation of the posterior.
            kl_beta: KL divergence weight.
            ll_scale: Log likelihood scaling factor.
            pretrain_epochs: Number of pretraining epochs for generative energy model,
                which can stabilize training, before switching to normal training regime
                that includes KL divergence.
            num_samples_test: Number of samples to use for test time predictions.
            stochastic_module_names: List of module names that should become density layers.
            freeze_backbone: If True, freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        super().__init__(
            model,
            loss_fn,
            prior_std,
            posterior_std_init,
            kl_beta,
            ll_scale,
            pretrain_epochs,
            num_samples_test,
            stochastic_module_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for the metrics."""
        # single output, MSE loss type case
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out[:, 0:1]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: input tensor
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            dictionary of predictions
        """
        with torch.no_grad():
            # squeeze the last dimension in case of 1 sample
            y_hat = torch.stack(
                [self.model(X) for _ in range(self.num_samples_test)], dim=-1
            ).squeeze(-1)

        return process_regression_prediction(y_hat)

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class DensityLayerModelClassification(DensityLayerModelBase):
    """Density Layer Model for Classification Tasks."""

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        task: str = "multiclass",
        prior_std: float = 0.1,
        posterior_std_init: float = 0.001,
        kl_beta: float = 1,
        ll_scale: float = 0.01,
        pretrain_epochs: int = 0,
        num_samples_test: int = 5,
        stochastic_module_names: list[int | str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a Density Layer Model for Classification Tasks.

        Args:
            model: PyTorch model that will be converted to a Density Layer Model.
            loss_fn: Loss function used for the target minimization, defaults to
                CrossEntropy Loss.
            task: Classification task type, one of "binary", "multiclass", or "multilabel".
            prior_std: Standard deviation of the prior.
            posterior_std_init: Initial standard deviation of the posterior.
            kl_beta: KL divergence weight.
            ll_scale: Log likelihood scaling factor.
            pretrain_epochs: Number of pretraining epochs for generative energy model,
                which can stabilize training, before switching to normal training regime
                that includes KL divergence.
            num_samples_test: Number of samples to use for test time predictions.
            stochastic_module_names: List of module names that should become density layers.
            freeze_backbone: If True, freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler

        Raises:
            AssertionError: If task is not one of the valid tasks.
        """
        assert task in self.valid_tasks
        self.task = task

        self.num_classes = _get_num_outputs(model)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        super().__init__(
            model,
            loss_fn,
            prior_std,
            posterior_std_init,
            kl_beta,
            ll_scale,
            pretrain_epochs,
            num_samples_test,
            stochastic_module_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for the metrics."""
        return out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: input tensor
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            dictionary of predictions
        """
        with torch.no_grad():
            # squeeze the last dimension in case of 1 sample
            y_hat = torch.stack(
                [self.model(X) for _ in range(self.num_samples_test)], dim=-1
            )
        return process_classification_prediction(y_hat)

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )
