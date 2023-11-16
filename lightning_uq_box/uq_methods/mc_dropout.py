"""Mc-Dropout module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .base import DeterministicModel
from .utils import (
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    process_regression_prediction,
    _get_num_outputs
)


class MCDropoutBase(DeterministicModel):
    """MC-Dropout Base class.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        num_mc_samples: int,
        loss_fn: nn.Module,
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model: pytorch model with dropout layers
            optimizer: optimizer used for training
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            lr_scheduler: learning rate scheduler
        """
        super().__init__(model, optimizer, loss_fn, lr_scheduler)

    def setup_task(self) -> None:
        """Setup task specific attributes."""
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

        def activate_dropout_recursive(model):
            for module in model.children():
                if isinstance(module, nn.Dropout):
                    module.train()
                elif isinstance(module, nn.Module):
                    activate_dropout_recursive(module)

        activate_dropout_recursive(self.model)


class MCDropoutRegression(MCDropoutBase):
    """MC-Dropout Model for Regression.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        num_mc_samples: int,
        loss_fn: nn.Module,
        burnin_epochs: int = 0,
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Regression.

        Args:
            model: pytorch model with dropout layers
            optimizer: optimizer used for training
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            burnin_epochs: number of burnin epochs before using the loss_fn
            lr_scheduler: learning rate scheduler
             from the predictive distribution
        """
        super().__init__(model, optimizer, num_mc_samples, loss_fn, lr_scheduler)
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def setup_task(self) -> None:
        """Setup task specific attributes."""
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
                [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        # TODO: this function is specific to regression
        # maybe the  name of this should be in the base class and be foreced to be overwritten by the subclasses?
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
        optimizer: type[Optimizer],
        num_mc_samples: int,
        loss_fn: nn.Module,
        task: str = "multiclass",
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Classification.

        Args:
            model: pytorch model with dropout layers
            optimizer: optimizer used for training
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            lr_scheduler: learning rate scheduler
        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = _get_num_outputs(model)
        super().__init__(model, optimizer, num_mc_samples, loss_fn, lr_scheduler)


        self.save_hyperparameters(ignore=["model", "loss_fn"])


    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics("val", self.task, self.num_classes)
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
#         optimizer: type[Optimizer],
#         num_mc_samples: int,
#         loss_fn: nn.Module,
#         burnin_epochs: int = 0,
#         lr_scheduler: type[LRScheduler] = None,
#     ) -> None:
#         super().__init__(
#             model, optimizer, num_mc_samples, loss_fn, burnin_epochs, lr_scheduler
#         )

#         self.save_hyperparameters(ignore=["model", "loss_fn"])


# class MCDropoutSegmentation(MCDropoutBase):
#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: type[Optimizer],
#         num_mc_samples: int,
#         loss_fn: nn.Module,
#         burnin_epochs: int = 0,
#         lr_scheduler: type[LRScheduler] = None,
#     ) -> None:
#         super().__init__(
#             model, optimizer, num_mc_samples, loss_fn, burnin_epochs, lr_scheduler
#         )

#         self.save_hyperparameters(ignore=["model", "loss_fn"])
