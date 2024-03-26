# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Implement Quantile Regression Model."""

import os
from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.eval_utils import compute_sample_mean_std_from_quantile

from .base import DeterministicModel
from .loss_functions import PinballLoss
from .utils import (
    default_px_regression_metrics,
    default_regression_metrics,
    freeze_segmentation_model,
    save_image_predictions,
    save_regression_predictions,
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

        if loss_fn is None:
            loss_fn = PinballLoss(quantiles=quantiles)

        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

        self.quantiles = quantiles
        self.median_index = self.quantiles.index(0.5)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")


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

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step."""
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key])

        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        return out_dict

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

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
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
            freeze_decoder: whether to freeze the decoder
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.freeze_decoder = freeze_decoder
        super().__init__(
            model, loss_fn, quantiles, freeze_backbone, optimizer, lr_scheduler
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        freeze_segmentation_model(self.model, self.freeze_backbone, self.freeze_decoder)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_px_regression_metrics("train")
        self.val_metrics = default_px_regression_metrics("val")
        self.test_metrics = default_px_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward`
                [batch_size x num_outputs x height x width]

        Returns:
            extracted mean used for metric computation [batch_size x 1 x height x width]
        """
        return out[
            :, self.median_index : self.median_index + 1, ...  # noqa: E203
        ].contiguous()

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()
        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        self.test_metrics(
            pred_dict["pred"].contiguous(), batch[self.target_key].squeeze()
        )

        return pred_dict

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
            "pred": self.adapt_output_for_metrics(out).squeeze(1),
            "lower": out[:, 0],
            "upper": out[:, -1],
        }

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
        save_image_predictions(outputs, batch_idx, self.pred_dir)
