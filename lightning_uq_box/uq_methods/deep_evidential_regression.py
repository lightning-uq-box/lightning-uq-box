# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Deep Evidential Regression."""

import os
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicModel
from .loss_functions import DERLoss
from .utils import (
    _get_num_outputs,
    default_px_regression_metrics,
    default_regression_metrics,
    freeze_segmentation_model,
    save_image_predictions,
    save_regression_predictions,
)


class DERLayer(nn.Module):
    """Deep Evidential Regression Layer.

    Taken from `here <https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py#L34>`_. # noqa: E501
    """

    def __init__(self):
        """Initialize a new Deep Evidential Regression Layer."""
        super().__init__()

    def forward(self, x):
        """Compute the DER parameters.

        Args:
            x: feature output from network [batch_size x 4]

        Returns:
            DER outputs of shape [batch_size x 4]
        """
        assert x.shape[1] == 4, "DER method expects 4 input features per sample."

        gamma = x[:, 0:1, ...]
        nu = nn.functional.softplus(x[:, 1:2, ...])
        alpha = nn.functional.softplus(x[:, 2:3, ...]) + 1.0
        beta = nn.functional.softplus(x[:, 3:4, ...])
        return torch.cat((gamma, nu, alpha, beta), dim=1)


class DER(DeterministicModel):
    """Deep Evidential Regression Model.

    Following the suggested implementation of
    the `Unreasonable Effectiveness of Deep Evidential Regression
    <https://github.com/pasteurlabs/unreasonable_effective_der/
    blob/4631afcde895bdc7d0927b2682224f9a8a181b2c/models.py#L22>`_

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2205.10060
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        coeff: float = 0.01,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            coeff: coefficient for the DER loss
                from the predictive distribution
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        # add DER Layer
        super().__init__(model, None, freeze_backbone, optimizer, lr_scheduler)

        self.save_hyperparameters(ignore=["model", "optimizer", "lr_scheduler"])

        # check that output is 4 dimensional
        assert _get_num_outputs(model) == 4, "DER model expects 4 outputs."

        # set DER Loss
        self.loss_fn = DERLoss(coeff)

        self.der_layer = DERLayer()

    def forward(self, X: Tensor) -> Any:
        """Forward pass of the model."""
        return self.der_layer(self.model(X))

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction Step Deep Evidential Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            dictionary with predictions and uncertainty measures
        """
        with torch.no_grad():
            pred = self.forward(X)  # [batch_size x 4 x othe_dims]

        gamma, nu, alpha, beta = (
            pred[:, 0:1, ...],
            pred[:, 1:2, ...],
            pred[:, 2:3, ...],
            pred[:, 3:4, ...],
        )

        epistemic_uct = self.compute_epistemic_uct(nu)
        aleatoric_uct = self.compute_aleatoric_uct(beta, alpha, nu)
        pred_uct = epistemic_uct + aleatoric_uct

        return {
            "pred": gamma,
            "pred_uct": pred_uct,
            "aleatoric_uct": aleatoric_uct,
            "epistemic_uct": epistemic_uct,
            "out": pred,
        }

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward` [batch_size x 4]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def compute_aleatoric_uct(self, beta: Tensor, alpha: Tensor, nu: Tensor) -> Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10 of the paper

        Args:
            beta: beta output DER model
            alpha: alpha output DER model
            nu: nu output DER model

        Returns:
            Aleatoric Uncertainty
        """
        return torch.sqrt(torch.div(beta * (1 + nu), alpha * nu))

    def compute_epistemic_uct(self, nu: Tensor) -> Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10: of the paper

        Args:
            nu: nu output DER model
        Returns:
            Epistemic Uncertainty
        """
        return torch.reciprocal(torch.sqrt(nu))

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


class DERPxRegression(DER):
    """Deep Evidential Regression Model for Pixelwise Regression with NLL."""

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        coeff: float = 0.01,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of the DER for Pixelwise Regression.

        Args:
            model: pytorch model
            coeff: coefficient for the DER loss
                from the predictive distribution
            freeze_backbone: whether to freeze the model backbone
            freeze_decoder: whether to freeze the model decoder
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.freeze_decoder = freeze_decoder
        super().__init__(model, coeff, freeze_backbone, optimizer, lr_scheduler)

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        # self.model[0] is the model without the DER Layer
        freeze_segmentation_model(self.model, self.freeze_backbone, self.freeze_decoder)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_px_regression_metrics("train")
        self.val_metrics = default_px_regression_metrics("val")
        self.test_metrics = default_px_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward` [batch_size x 4]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1, ...].contiguous()

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir):
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
        save_image_predictions(outputs, batch_idx, self.pred_dir)
