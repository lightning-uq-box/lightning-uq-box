# MIT License
# Copyright (c) 2024 CVLAB @ EPFL

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
# Adapted from ...


"""ZigZag Universal Sampling-free Uncertainty Estimation."""

import os

import torch
import torch.nn as nn
from einops import repeat
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicModel
from .utils import (
    _get_input_layer_name_and_module,
    default_regression_metrics,
    save_regression_predictions,
)


class ZigZagBase(DeterministicModel):
    """ZigZag Uncertainty Quantification Base.

    If you use this method in your work, please cite:

    * https://openreview.net/forum?id=QSvb6jBXML
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        blank_const: int = -100,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instance of ZigZag.

        Args:
            model: PyTorch model.
            loss_fn: Loss function.
            blank_const: constant for the blank zig zag input, should be a
                value far from training targets
            freeze_backbone: Whether or not to freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.

        """
        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)
        self.blank_const = blank_const
        self.check_input_layer()

    def check_input_layer(self) -> None:
        """Check whether the input layer is linear or convolutional.

        This has an effect on how the inputs are concatenated for ZigZag.
        """
        _, module = _get_input_layer_name_and_module(self.model)

        if isinstance(module, nn.Linear):
            self.input_linear = True
        else:
            self.input_linear = False

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass of Zig Zag method.

        Args:
            x: Input tensor.
            y: Target tensor.

        Returns:
            Output of model with Zig Zag operation.
        """
        # create additional feature dimension either of blanks or targets
        if y is None:
            if self.input_linear:
                x_in = torch.concat(
                    [x, self.blank_const * torch.ones([x.shape[0], 1])], dim=1
                )
            else:
                x_in = torch.cat([x, self.blank_const * torch.ones_like(x)], dim=1)
        else:
            if self.input_linear:
                # Y_t = torch.cat([self.blank_const * torch.ones_like(y), y])
                # try:
                x_in = torch.concat([x, y], dim=1)
            else:
                inputs_1 = torch.cat(
                    [x, self.blank_const * torch.ones_like(x)], dim=1
                ).cpu()

                # The second input with actual targets, the second term in Eq. 1
                t_inputs = y.reshape(-1, 1, 1, 1) * torch.ones_like(x)
                inputs_2 = torch.cat([x, t_inputs], dim=1).cpu()

                # For simplicity, we randomly choose which inputs to use for computing the first or second terms
                # Could compute the whole Eq. 1 loss here instead
                p = 0.5
                mask = (
                    (torch.empty(inputs_1.shape[0], 1, 1, 1).uniform_(0, 1) > p)
                    .float()
                    .cpu()
                )
                x_in = inputs_1 * mask + inputs_2 * (1 - mask)

        return self.model(x_in)

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
        X_repeat = repeat(X, "b ... -> (repeat b) ...", repeat=2)
        if self.input_linear:
            y_in = torch.cat([self.blank_const * torch.ones_like(y), y])
        else:
            y_in = y

        out = self.forward(X_repeat, y_in)
        loss = self.loss_fn(out, repeat(y, "b ... -> (repeat b) ...", repeat=2))

        # compute metrics only on the real input not the zigzag condition
        if X.shape[0] > 1:
            self.train_metrics(out[: X.shape[0]], y)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]
        X_repeat = repeat(X, "b ... -> (repeat b) ...", repeat=2)

        if self.input_linear:
            y_in = torch.cat([self.blank_const * torch.ones_like(y), y])
        else:
            y_in = y

        out = self.forward(X_repeat, y_in)
        loss = self.loss_fn(out, repeat(y, "b ... -> (repeat b) ...", repeat=2))

        # compute metrics only on the real input not the zigzag condition
        if X.shape[0] > 1:
            self.val_metrics(out[: X.shape[0]], y)

        self.log("train_loss", loss)
        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step."""
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(pred_dict["pred"], batch[self.target_key])

        pred_dict["pred"] = pred_dict["pred"].detach().cpu().squeeze(-1)

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        return pred_dict

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
        with torch.no_grad():
            Y_1 = self.forward(X)
            Y_2 = self.forward(X, Y_1)

        return {"pred": Y_1, "pred_uct": torch.linalg.norm(Y_1 - Y_2, dim=1)}


class ZigZagRegression(ZigZagBase):
    """Zig Zag Uncertainty Estimation for Regression.

    If you use this method in your work, please cite:

    * https://openreview.net/forum?id=QSvb6jBXML
    """

    pred_file_name = "preds.csv"

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

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
