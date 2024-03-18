# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Implement Quantile Regression Model."""

import os
from typing import Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.eval_utils import compute_sample_mean_std_from_quantile

from .base import DeterministicModel
from .loss_functions import PinballLoss
from .utils import (
    _get_num_outputs,
    default_regression_metrics,
    freeze_model_backbone,
    save_regression_predictions,
    default_px_regression_metrics,
)


class QuantileRegressionBase(DeterministicModel):
    """Quantile Regression Base Module.

    If you use this model in your research, please cite the following paper:

    * https://www.jstor.org/stable/1913643
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Quantile Regression Model.

        Args:
            model: pytorch model
            loss_fn: loss function
            quantiles: quantiles to compute
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        assert all(i < 1 for i in quantiles), "Quantiles should be less than 1."
        assert all(i > 0 for i in quantiles), "Quantiles should be greater than 0."
        assert _get_num_outputs(model) == len(
            quantiles
        ), "The num of desired quantiles should match num_outputs of the model."

        if loss_fn is None:
            loss_fn = PinballLoss(quantiles=quantiles)

        self.freeze_backbone = freeze_backbone

        super().__init__(model, loss_fn, optimizer, lr_scheduler)

        self.quantiles = quantiles
        self.median_index = self.quantiles.index(0.5)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")
        self.freeze_model()

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            freeze_model_backbone(self.model)


class QuantileRegression(QuantileRegressionBase):
    """Quantile Regression Module for Regression.

    If you use this model in your research, please cite the following paper:

    * https://www.jstor.org/stable/1913643
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Quantile Regression Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            loss_fn: loss function
            quantiles: quantiles to compute
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            model, loss_fn, quantiles, freeze_backbone, optimizer, lr_scheduler
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, self.median_index : self.median_index + 1]  # noqa: E203

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Quantile Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            predicted uncertainties
        """
        with torch.no_grad():
            out = self.model(X)  # [batch_size, len(self.quantiles)]

        median = self.adapt_output_for_metrics(out)
        _, std = compute_sample_mean_std_from_quantile(out, self.hparams.quantiles)

        return {
            "pred": median,
            "pred_uct": std,
            "lower_quant": out[:, 0],
            "upper_quant": out[:, -1],
            "aleatoric_uct": std,
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
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class QuantilePxRegression(QuantileRegressionBase):
    """Quantile Regression for Pixelwise Regression."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Quantile Regression Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            loss_fn: loss function
            quantiles: quantiles to compute
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            model, loss_fn, quantiles, freeze_backbone, optimizer, lr_scheduler
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_px_regression_metrics("train")
        self.val_metrics = default_px_regression_metrics("val")
        self.test_metrics = default_px_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs x height x width]

        Returns:
            extracted mean used for metric computation [batch_size x 1 x height x width]
        """
        return out[:, self.median_index: self.median_index+1, ...].contiguous()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Quantile Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            predicted uncertainties
        """
        with torch.no_grad():
            out = self.model(X)  # [batch_size, len(self.quantiles)]

        return {
            "pred": self.adapt_output_for_metrics(out),
            "lower_quant": out[:, 0],
            "upper_quant": out[:, -1],
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
        pass
