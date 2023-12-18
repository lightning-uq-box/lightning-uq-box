# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Conformal Risk Control."""

from typing import Optional, Union

import numpy as np
import torch
from lightning import LightningModule
from scipy.optimize import brentq
from torch import Tensor
from torch.nn.modules import Module

from lightning_uq_box.uq_methods import PosthocBase

from .utils import default_segmentation_metrics


class ConformalRiskControl(PosthocBase):
    """Conformal Risk Control.

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2208.02814
    """

    pred_file_name = "preds.csv"
    def __init__(
        self,
        model: Union[LightningModule, Module],
        lamda_param: Optional[int] = None,
        alpha: float = 0.1,
    ) -> None:
        """Initialize ConformalRiskControl.

        Args:
            model: A lightning or torch module
            lamda_param: The lamda parameter for the conformal risk control method. If None, it will be optimized for via
                Brent's method as implemented in scipy
            alpha: 1 - alpha is the desired coverage
        """
        super().__init__(model)

        self.lamda_param = lamda_param
        self.alpha = alpha

        self.setup_task()

    def on_validation_end(self) -> None:
        """Apply Conformal Risk Control conformal method."""
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach().cpu().numpy()

        if all_logits.dim() == 4:
            all_logits = all_logits.squeeze(1)
        all_sigmoid = all_logits.sigmoid().cpu().numpy()

        n = all_logits.shape[0]
    
        def false_negative_rate(pred_masks: np.ndarray, true_masks: np.ndarray):
            """Compute the false negative rate.

            Args:
                pred_masks: Predicted masks
                true_masks: True masks
            """
            return (
                1
                - (
                    (pred_masks * true_masks).sum(axis=1).sum(axis=1)
                    / true_masks.sum(axis=1).sum(axis=1)
                ).mean()
            )

        def lamhat_threshold(lam):
            return false_negative_rate(all_sigmoid >= lam, all_labels) - (
                (n + 1) / n * self.alpha - 1 / (n + 1)
            )  

        self.lamhat = brentq(lamhat_threshold, 0, 1)

        self.post_hoc_fitted = True

    def adjust_model_logits(self, model_logits: torch.Tensor) -> torch.Tensor:
        """Adjust the model logits with the lamhat threshold.

        Args:
            model_logits: The model logits

        Returns:
            The adjusted model logits
        """
        mask = model_logits >= self.lamhat
        adjusted_logits = torch.where(mask, model_logits, 0)
        return adjusted_logits


class ConformalRiskControlSegmentation(ConformalRiskControl):
    """Conformal Risk Control for Segmentation."""

    def setup_task(self) -> None:
        """Setup the task."""
        self.test_metrics = default_segmentation_metrics(prefix="test", num_classes=2, task="binary")

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology.

        Args:
            batch: batch of shape [batch_size x num_channels x height x width]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            logits and CRS adjusted predictions
        """
        # need to set manually because of inference mode
        self.eval()
        pred_dict = self.predict_step(batch[self.input_key])

        # logging metrics
        self.test_metrics(pred_dict["pred"].flatten(), batch[self.target_key].flatten())
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Conformal Risk Control applied.

        Args:
            X: prediction batch of shape [batch_size x num_channels x height x width]

        Returns:
            logits and conformalized prediction sets
        """
        # need to set manually because of inference mode
        self.eval()
        with torch.no_grad():
            if hasattr(self.model, "predict_step"):
                logits = self.model.predict_step(X)["logits"]
            else:
                logits = self.model(X)
            output = self.adjust_model_logits(logits)

        return {"pred": output, "logits": logits}
