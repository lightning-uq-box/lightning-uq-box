# MIT License
# Copyright (c) 2024 CVLAB @ EPFL

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
# Adapted from https://github.com/cvlab-epfl/zigzag as a LightningModule


"""ZigZag Universal Sampling-free Uncertainty Estimation."""

import os

import torch
import torch.nn as nn
from einops import repeat
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam as Adam

from .base import DeterministicModel
from .utils import (
    _get_input_layer_name_and_module,
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    save_classification_predictions,
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

    def forward(
        self, x: Tensor, y: Tensor | None = None, training: bool = False
    ) -> Tensor:
        """Forward pass of Zig Zag method.

        Args:
            x: Input tensor.
            y: Target tensor.
            training: Whether or not the model is in training mode,
                which affects the Zig Zag operation for conv input layers

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
                batch_size, _, height, width = x.shape
                ones_tensor = torch.ones(
                    [batch_size, 1, height, width], device=x.device, dtype=x.dtype
                )
                x_in = torch.cat([x, self.blank_const * ones_tensor], dim=1)
        else:
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            if self.input_linear:
                # classification labels are just 1D
                x_in = torch.concat([x, torch.atleast_2d(y)], dim=1)
            else:
                batch_size, _, height, width = x.shape
                channel_y = torch.atleast_2d(y).shape[-1]
                ones_tensor = torch.ones(
                    [batch_size, channel_y, height, width],
                    device=x.device,
                    dtype=x.dtype,
                )
                if training:
                    inputs_1 = torch.cat([x, self.blank_const * ones_tensor], dim=1)
                    # The second input with actual targets, the second term in Eq. 1
                    t_inputs = y.reshape(-1, 1, 1, 1) * ones_tensor
                    inputs_2 = torch.cat([x, t_inputs], dim=1)

                    p = 0.5
                    mask = (
                        (torch.empty(inputs_1.shape[0], 1, 1, 1).uniform_(0, 1) > p)
                        .float()
                        .to(x.device)
                    )
                    x_in = inputs_1 * mask + inputs_2 * (1 - mask)
                else:
                    x_in = torch.cat([x, y.reshape(-1, 1, 1, 1) * ones_tensor], dim=1)

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
        if self.input_linear:
            x_in = repeat(X, "b ... -> (repeat b) ...", repeat=2)
            y_in = torch.cat([self.blank_const * torch.ones_like(y), y])
            y_target = repeat(y, "b ... -> (repeat b) ...", repeat=2)
        else:
            x_in = X
            y_in = y
            y_target = y

        out = self.forward(x_in, y_in, training=True)
        loss = self.loss_fn(out, y_target)

        # compute metrics only on the real input not the zigzag condition
        if X.shape[0] > 1:
            if self.input_linear:
                self.train_metrics(out[: X.shape[0]], y)
            else:
                self.train_metrics(out, y)

        self.log("train_loss", loss, batch_size=X.shape[0])
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
        if self.input_linear:
            x_in = repeat(X, "b ... -> (repeat b) ...", repeat=2)
            y_in = torch.cat([self.blank_const * torch.ones_like(y), y])
            y_target = repeat(y, "b ... -> (repeat b) ...", repeat=2)
        else:
            x_in = X
            y_in = y
            y_target = y

        out = self.forward(x_in, y_in, training=False)
        loss = self.loss_fn(out, y_target)

        # compute metrics only on the real input not the zigzag condition
        if X.shape[0] > 1:
            if self.input_linear:
                self.val_metrics(out[: X.shape[0]], y)
            else:
                self.val_metrics(out, y)

        self.log("val_loss", loss, batch_size=X.shape[0])
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
            Y_1 = self.forward(X, training=False)
            Y_2 = self.forward(X, Y_1, training=False)
        return {"pred": Y_1, "pred_uct": torch.linalg.norm(Y_1 - Y_2, dim=1)}

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


class ZigZagClassification(ZigZagBase):
    """Zig Zag Uncertainty Estimation for Classification.

    If you use this method in your work, please cite:

    * https://openreview.net/forum?id=QSvb6jBXML

    """

    pred_file_name = "preds.csv"

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        blank_const: int = -100,
        task: str = "multiclass",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instance of ZigZag for classification.

        Args:
            model: PyTorch model.
            loss_fn: Loss function.
            blank_const: constant for the blank zig zag input, should be a
                value far from training targets
            task: Task type. One of "binary", "multiclass", "multilabel".
            freeze_backbone: Whether or not to freeze the backbone.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.
        """
        self.num_classes = _get_num_outputs(model)
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        super().__init__(
            model, loss_fn, blank_const, freeze_backbone, optimizer, lr_scheduler
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

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            prediction dictionary
        """
        self.model.eval()
        with torch.no_grad():
            Y_1 = self.forward(X, training=False)
            Y_1_softmax = torch.softmax(Y_1, dim=1)
            Y_1_labels = torch.argmax(Y_1_softmax, dim=1)
            Y_2 = self.forward(X, Y_1_labels, training=False)
            Y_2_softmax = torch.softmax(Y_2, dim=1)

        return {
            "pred": Y_1_softmax,
            "pred_uct": torch.abs(Y_1_softmax - Y_2_softmax).mean(dim=1),
            "logits": Y_1,
        }

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
