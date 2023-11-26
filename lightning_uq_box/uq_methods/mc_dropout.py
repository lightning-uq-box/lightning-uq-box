# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Mc-Dropout module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicModel
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    process_regression_prediction,
)


def find_dropout_layers(model: nn.Module) -> list[str]:
    """Find dropout layers in model."""
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append(name)

    # if not dropout_layers:
    #     raise UserWarning(
    #         (
    #           "No dropout layers found in model, maybe dropout "
    #           "is implemented through nn.fucntional?"
    #         )
    #     )
    return dropout_layers


class MCDropoutBase(DeterministicModel):
    """MC-Dropout Base class.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        dropout_layer_names: list[str] = [],
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            dropout_layer_names: names of dropout layers to activate during prediction
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(model, loss_fn, optimizer, lr_scheduler)

        if not dropout_layer_names:
            dropout_layer_names = find_dropout_layers(model)
        self.dropout_layer_names = dropout_layer_names

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        pass

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        out = self.forward(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), batch[self.target_key])

        return loss

    def activate_dropout(self) -> None:
        """Activate dropout layers."""

        def activate_dropout_recursive(model, prefix=""):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in self.dropout_layer_names and isinstance(
                    module, nn.Dropout
                ):
                    module.train()
                elif isinstance(module, nn.Module):
                    activate_dropout_recursive(module, full_name)

        activate_dropout_recursive(self.model)


class MCDropoutRegression(MCDropoutBase):
    """MC-Dropout Model for Regression.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        burnin_epochs: int = 0,
        dropout_layer_names: list[str] = [],
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Regression.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            burnin_epochs: number of burnin epochs before using the loss_fn
            dropout_layer_names: names of dropout layers to activate during prediction
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
                from the predictive distribution
        """
        super().__init__(
            model, num_mc_samples, loss_fn, dropout_layer_names, optimizer, lr_scheduler
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model."""
        assert out.shape[-1] <= 2, "Ony support single mean or Gaussian output."
        return out[:, 0:1]

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        out = self.forward(batch[self.input_key])

        if self.current_epoch < self.hparams.burnin_epochs:
            loss = nn.functional.mse_loss(
                self.extract_mean_output(out), batch[self.target_key]
            )
        else:
            loss = self.loss_fn(out, batch[self.target_key])

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), batch[self.target_key])

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()  # activate dropout during prediction
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_regression_prediction(preds)


class MCDropoutClassification(MCDropoutBase):
    """MC-Dropout Model for Classification.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        task: str = "multiclass",
        dropout_layer_names: list[str] = [],
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Classification.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            dropout_layer_names: names of dropout layers to activate during prediction
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = _get_num_outputs(model)
        super().__init__(
            model, num_mc_samples, loss_fn, dropout_layer_names, optimizer, lr_scheduler
        )

        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
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

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model."""
        return out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()  # activate dropout during prediction
        with torch.no_grad():
            preds = torch.stack(
                [
                    F.softmax(self.model(X), dim=1)
                    for _ in range(self.hparams.num_mc_samples)
                ],
                dim=-1,
            )  # shape [batch_size, num_outputs, num_samples]

        return process_classification_prediction(preds)


# class MCDropoutPxRegression(MCDropoutBase):
#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: OptimizerCallable = torch.optim.Adam,
#         num_mc_samples: int,
#         loss_fn: nn.Module,
#         burnin_epochs: int = 0,
#         lr_scheduler: LRSchedulerCallable = None,
#     ) -> None:
#         super().__init__(
#             model, optimizer, num_mc_samples, loss_fn, burnin_epochs, lr_scheduler
#         )

#         self.save_hyperparameters(ignore=["model", "loss_fn"])


# class MCDropoutSegmentation(MCDropoutBase):
#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: OptimizerCallable = torch.optim.Adam,
#         num_mc_samples: int,
#         loss_fn: nn.Module,
#         burnin_epochs: int = 0,
#         lr_scheduler: LRSchedulerCallable = None,
#     ) -> None:
#         super().__init__(
#             model, optimizer, num_mc_samples, loss_fn, burnin_epochs, lr_scheduler
#         )

#         self.save_hyperparameters(ignore=["model", "loss_fn"])
