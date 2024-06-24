# MIT License
# Copyright (c) 2024 CVLAB @ EPFL

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
# Adapted from https://github.com/cvlab-epfl/iter_unc as a LightningModule


"""Enabling Uncertainty Estimation in Iterative Neural Networks."""

import os

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam as Adam

from .utils import default_regression_metrics, save_regression_predictions
from .zigzag import ZigZagBase


class IterativeUncertaintyBase(ZigZagBase):
    """Base class for Iterative Uncertainty Estimation.

    If you use this method in your work, please cite:

    * https://arxiv.org/abs/2403.16732
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_iter: int = 3,
        blank_const: int = -100,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize Iterative Uncertainty Base Class.

        Args:
            model: PyTorch model.
            loss_fn: Loss function.
            blank_const: constant for the blank zig zag input, should be a
                value far from training targets
            num_iter: Number of iterations for iterative uncertainty procedure
            freeze_backbone: Whether or not to freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.

        Raises:
            AssertionError: If the number of iterations is less than 1.
        """
        assert num_iter > 1, "Number of iterations should be greater than 1"
        self.num_iter = num_iter
        super().__init__(
            model, loss_fn, blank_const, freeze_backbone, optimizer, lr_scheduler
        )

    def compute_loss(
        self, x_in: Tensor, y_in: Tensor, target: Tensor, training: bool = False
    ) -> Tensor:
        """Compute loss for iterative uncertainty estimation.

        Args:
            x_in: Input tensor.
            y_in: y input tensor for Zig Zag step.
            target: Target tensor.
            training: Whether or not the model is in training mode,
                which affects the Zig Zag operation for conv input layers
        """
        Y_t = self.forward(x_in, y_in, training)
        loss = self.loss_fn(Y_t, target)

        for _ in range(self.num_iter):
            Y_t = self.forward(x_in, Y_t.detach())
            loss += self.loss_fn(Y_t, target)

        return Y_t, loss


class IterativeUncertaintyRegression(IterativeUncertaintyBase):
    """Iterative Uncertainty Estimation for Regression.

    If you use this method in your work, please cite:

    * https://arxiv.org/abs/2403.16732
    """

    pred_file_name = "preds.csv"

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict Step.

        Conducts two forward passes. One with the input, and
        a second one with the input and the output of the first
        forward pass.

        Args:
            X: prediction input tensor
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            prediction dictionary
        """
        preds: list[Tensor] = []
        with torch.no_grad():
            Y_t = self.forward(X, training=False)
            preds.append(Y_t)
            for _ in range(self.num_iter):
                Y_t = self.forward(X, Y_t, training=False)
                preds.append(Y_t)

            preds = torch.stack(preds)

        return {"pred": Y_t, "pred_uct": torch.std(preds, dim=0)}

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
