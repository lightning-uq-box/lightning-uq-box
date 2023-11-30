# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Deterministic Model that predicts parameters of Gaussian."""

import os

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicModel
from .loss_functions import NLL
from .utils import default_regression_metrics, save_regression_predictions


class MVEBase(DeterministicModel):
    """Mean Variance Estimation Network Base Class.

    If you use this model in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/374138
    """

    def __init__(
        self,
        model: nn.Module,
        burnin_epochs: int,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model.

        Args:
            model: pytorch model
            burnin_epochs: number of burnin epochs before switiching to NLL
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(model, None, optimizer, lr_scheduler)

        self.loss_fn = NLL()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

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
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )
        else:
            loss = self.loss_fn(out, batch[self.target_key])

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss


class MVERegression(MVEBase):
    """Mean Variance Estimation Model for Regression that is trained with NLL.

    If you use this model in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/374138
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        burnin_epochs: int,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Mean Variance Estimation Model for Regression.

        Args:
            model: pytorch model
            burnin_epochs: number of burnin epochs before switiching to NLL
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler

        """
        super().__init__(model, burnin_epochs, optimizer, lr_scheduler)
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation."""
        assert out.shape[-1] <= 2, "Gaussian output."
        return out[:, 0:1]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            preds = self.model(X)

        mean, log_sigma_2 = preds[:, 0], preds[:, 1].cpu()
        eps = torch.ones_like(log_sigma_2) * 1e-6
        std = torch.sqrt(eps + np.exp(log_sigma_2))

        return {"pred": mean, "pred_uct": std, "aleatoric_uct": std, "out": preds}

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
