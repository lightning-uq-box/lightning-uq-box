# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Deterministic Model that predicts parameters of Gaussian."""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicPixelRegression, DeterministicRegression
from .loss_functions import NLL
from .utils import save_image_predictions, save_regression_predictions


class MVEBase(DeterministicRegression):
    """Mean Variance Estimation Network Base Class.

    If you use this model in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/374138
    """

    def __init__(
        self,
        model: nn.Module,
        burnin_epochs: int,
        n_targets: int = 1,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model.

        Args:
            model: pytorch model
            burnin_epochs: number of burnin epochs before switiching to NLL
            n_targets: number of regression targets
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            model, NLL(), n_targets, freeze_backbone, optimizer, lr_scheduler
        )

        self.burnin_epochs = burnin_epochs

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
        out = self.forward(batch[self.input_key])

        if self.current_epoch < self.burnin_epochs:
            loss = nn.functional.mse_loss(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )
        else:
            loss = self.loss_fn(out, batch[self.target_key])

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
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
        n_targets: int = 1,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Mean Variance Estimation Model for Regression.

        Args:
            model: pytorch model
            burnin_epochs: number of burnin epochs before switiching to NLL
            n_targets: number of regression targets
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler

        """
        super().__init__(
            model, burnin_epochs, n_targets, freeze_backbone, optimizer, lr_scheduler
        )

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

        mean, log_sigma_2 = preds[:, 0:1], preds[:, 1:2].cpu()
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


class MVEPxRegression(DeterministicPixelRegression):
    """Mean Variance Estimation Model for Pixelwise Regression with NLL."""

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        n_targets: int = 1,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of MVE for Pixelwise Regression.

        Args:
            model: pytorch model
            n_targets: number of regression targets
            freeze_backbone: whether to freeze the backbone
            freeze_decoder: whether to freeze the decoder
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        super().__init__(
            model,
            NLL(),
            n_targets,
            freeze_backbone,
            freeze_decoder,
            optimizer,
            lr_scheduler,
        )
        self.save_preds = save_preds

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        assert out.shape[1] <= 2, "Gaussian output."
        return out[:, 0:1, ...].contiguous()

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir) and self.save_preds:
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)
