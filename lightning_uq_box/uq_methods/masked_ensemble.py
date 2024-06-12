# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Masked Ensemble Model."""

import os
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.models.masked_ensemble.utils import (
    convert_deterministic_to_masked_ensemble,
)

from .base import BaseModule
from .loss_functions import NLL
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    process_regression_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


class MasksemblesBase(BaseModule):
    """Base class for Masked Ensemble models.

    If you use this model in your work, please cit:

    * https://arxiv.org/abs/2012.08334

    The input from the dataloader will be repeated for each estimator, so
    consider this when defining the batch size regarding memory usage.

    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_estimators: int,
        scale: float,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize Masked Ensemble model.

        Args:
            model: PyTorch model to turn into a Masked Ensemble and train it.
            loss_fn: Loss function.
            num_estimators: The number of estimators (masks) to generate.
            scale: The scale factor for mask generation. Muste be a scaler in
                the interval [1, 6].
            optimizer: Optimizer to use.
            lr_scheduler: Learning rate scheduler.
        """
        super().__init__()
        self.num_estimators = num_estimators
        self.scale = scale

        self.loss_fn = loss_fn

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.convert_to_masked_ensemble(model)

        self.model = model

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        raise NotImplementedError

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for the metrics.

        Args:
            out: model output

        Returns:
            adapted output
        """
        if isinstance(self.loss_fn, nn.MSELoss):
            return out
        elif isinstance(self.loss_fn, NLL):
            return out[:, 0:1]

    def convert_to_masked_ensemble(self, model: nn.Module) -> None:
        """Convert model to a Masked Ensemble model.

        Args:
            model: PyTorch model to turn into a Masked Ensemble
        """
        convert_deterministic_to_masked_ensemble(
            model, num_estimators=self.num_estimators, scale=self.scale
        )

    def forward(self, x: Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, *input_shape]

        Returns:
            Output tensor of shape [batch_size * num_estimators, *output_shape]
        """
        # repeat the input tensor for each estimator
        x = repeat(x, "b ... -> (n b) ...", n=self.num_estimators)
        return self.model(x)

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
        y_hat = self.forward(X)
        y_repeat = repeat(y, "b ... -> (n b) ...", n=self.num_estimators)
        loss = self.loss_fn(y_hat, y_repeat)

        self.log("train_loss", loss, batch_size=X.size(0) * self.num_estimators)
        self.train_metrics(self.adapt_output_for_metrics(y_hat), y_repeat)

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

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

        y_hat = self.forward(X)
        y_repeat = repeat(y, "b ... -> (n b) ...", n=self.num_estimators)
        loss = self.loss_fn(y_hat, y_repeat)

        self.log("val_loss", loss, batch_size=X.size(0) * self.num_estimators)
        self.val_metrics(self.adapt_output_for_metrics(y_hat), y_repeat)

        return loss

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Compute and return the test prediction.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            test prediction with uncertainty
        """
        pred_dict = self.predict_step(batch[self.input_key])

        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        # compute test metric with ensemble mean
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(pred_dict["pred"]), batch[self.target_key]
            )

        pred_dict["pred"] = pred_dict["pred"].detach().cpu().squeeze(-1)

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        return pred_dict

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class MasksemblesRegression(MasksemblesBase):
    """Masked Ensemble for regression tasks.

    If you use this model in your work, please cit:

    * https://arxiv.org/abs/2012.08334

    The input from the dataloader will be repeated for each estimator, so
    consider this when defining the batch size regarding memory usage.
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_estimators: int,
        scale: float,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize Masked Ensemble for regression tasks.

        Args:
            model: PyTorch model to turn into a Masked Ensemble and train it.
            loss_fn: Loss function to train the model.
            num_estimators: The number of estimators (masks) to generate.
            scale: The scale factor for mask generation. Muste be a scaler in
                the interval [1, 6].
            optimizer: Optimizer to use.
            lr_scheduler: Learning rate scheduler.
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            num_estimators=num_estimators,
            scale=scale,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def predict_step(self, x: Tensor) -> Tensor:
        """Predict the output of the model.

        Args:
            x: Input tensor of shape [batch_size, *input_shape]

        Returns:
            Output tensor of shape [batch_size, *output_shape]
        """
        with torch.no_grad():
            ensemble_pred = self.forward(x)

        # rearange to put the estimators (samples) in the last dimension
        preds = rearrange(ensemble_pred, "(n b) ... -> b ... n", n=self.num_estimators)

        return process_regression_prediction(preds)

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


class MasksemblesClassification(MasksemblesBase):
    """Masked Ensemble for classification tasks.

    If you use this model in your work, please cit:

    * https://arxiv.org/abs/2012.08334

    The input from the dataloader will be repeated for each estimator, so
    consider this when defining the batch size regarding memory usage.
    """

    pred_file_name = "preds.csv"

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_estimators: int,
        scale: float,
        task: str = "multiclass",
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ):
        """Initialize Masked Ensemble for classification tasks.

        Args:
            model: PyTorch model to turn into a Masked Ensemble and train it.
            loss_fn: Loss function to train the model.
            num_estimators: The number of estimators (masks) to generate.
            scale: The scale factor for mask generation. Muste be a scaler in
                the interval [1, 6].
            task: what kind of classification task, choose one of
                ["binary", "multiclass", "multilabel"]
            optimizer: Optimizer to use.
            lr_scheduler: Learning rate scheduler.
        """
        self.num_classes = _get_num_outputs(model)
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            num_estimators=num_estimators,
            scale=scale,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
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
        """Adapt the output for the metrics.

        Args:
            out: model output

        Returns:
            adapted output
        """
        return out

    def predict_step(self, x: Tensor) -> Tensor:
        """Predict the output of the model.

        Args:
            x: Input tensor of shape [batch_size, *input_shape]

        Returns:
            Output tensor of shape [batch_size, *output_shape]
        """
        with torch.no_grad():
            ensemble_pred = self.forward(x)

        # rearange to put the estimators (samples) in the last dimension
        preds = rearrange(ensemble_pred, "(n b) ... -> b ... n", n=self.num_estimators)

        return process_classification_prediction(preds)

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
