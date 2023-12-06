# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Deep Evidential Regression."""

import os

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import DeterministicModel
from .loss_functions import DERLoss
from .utils import (
    _get_num_outputs,
    default_regression_metrics,
    save_regression_predictions,
)


class DERLayer(nn.Module):
    """Deep Evidential Regression Layer.

    Taken from `here <https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py#L34>`_. # noqa: E501
    """

    def __init__(self):
        """Initialize a new Deep Evidential Regression Layer."""
        super().__init__()
        self.in_features = 4
        self.out_features = 4

    def forward(self, x):
        """Compute the DER parameters.

        Args:
            x: feature output from network [batch_size x 4]

        Returns:
            DER outputs of shape [batch_size x 4]
        """
        assert x.dim() == 2, "Input X should be 2D."
        assert x.shape[-1] == 4, "DER method expects 4 inputs per sample."

        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


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
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            coeff: coefficient for the DER loss
                from the predictive distribution
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__(model, None, optimizer, lr_scheduler)

        self.save_hyperparameters(ignore=["model", "optimizer", "lr_scheduler"])

        # check that output is 4 dimensional
        assert _get_num_outputs(model) == 4, "DER model expects 4 outputs."

        # add DER Layer
        self.model = nn.Sequential(self.model, DERLayer())

        # set DER Loss
        # TODO need to give control over the coeff through config or argument
        self.loss_fn = DERLoss(coeff)

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

        Returns:
            dictionary with predictions and uncertainty measures
        """
        with torch.no_grad():
            pred = self.model(X)  # [batch_size x 4]
            pred_np = pred.cpu().numpy()

        gamma, nu, alpha, beta = (
            pred[:, 0:1],
            pred_np[:, 1],
            pred_np[:, 2],
            pred_np[:, 3],
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

    def compute_aleatoric_uct(
        self, beta: np.ndarray, alpha: np.ndarray, nu: np.ndarray
    ) -> Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10:

        Args:
            beta: beta output DER model
            alpha: alpha output DER model
            nu: nu output DER model

        Returns:
            Aleatoric Uncertainty
        """
        # Equation 10 from the above paper
        return np.sqrt(np.divide(beta * (1 + nu), alpha * nu))

    def compute_epistemic_uct(self, nu: np.ndarray) -> np.ndarray:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10:

        Args:
            nu: nu output DER model
        Returns:
            Epistemic Uncertainty
        """
        return np.reciprocal(np.sqrt(nu))

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
