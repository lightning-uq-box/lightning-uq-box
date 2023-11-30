# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""conformalized Quantile Regression Model."""

import math
import os
from typing import Dict, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from .base import PosthocBase
from .utils import save_regression_predictions


def compute_q_hat_with_cqr(
    cal_preds: Tensor, cal_labels: Tensor, alpha: float
) -> float:
    """Compute q_hat which is the adjustment factor for quantiles.

    Check trusted computation here.

    Args:
        cal_preds: calibration set predictions
        cal_labels: calibration set targets
        alpha: 1 - alpha is desired error rate for quantile

    Returns:
        q_hat the computed quantile by which prediction intervals
        can be adjusted according to cqr
    """
    cal_labels = cal_labels.squeeze()

    n = cal_labels.shape[0]
    cal_upper = cal_preds[:, -1]
    cal_lower = cal_preds[:, 0]

    # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
    cal_scores = torch.maximum(cal_labels - cal_upper, cal_lower - cal_labels)

    # Get the score quantile
    q_hat = torch.quantile(
        cal_scores, math.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    return q_hat


class ConformalQR(PosthocBase):
    """Conformalized Quantile Regression.

    If you use this model, please cite the following paper:

    * https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html # noqa: E501
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: Union[nn.Module, LightningModule],
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new CQR model.

        Args:
            model: underlying model to be wrapped
            quantiles: quantiles to be used for CQR
        """
        super().__init__(model)

        self.save_hyperparameters(ignore=["model"])

        self.quantiles = quantiles

        self.alpha = min(self.hparams.quantiles)

        self.desired_coverage = 1 - self.alpha  # 1-alpha is the desired coverage

    def adjust_model_logits(self, model_output: Tensor) -> Tensor:
        """Conformalize underlying model output.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            conformalized model predictions
        """
        # conformalize predictions
        cqr_sets = torch.stack(
            [
                model_output[:, 0] - self.q_hat,
                model_output[:, 1],
                model_output[:, -1] + self.q_hat,
            ],
            dim=1,
        )
        return cqr_sets

    def on_validation_epoch_end(self) -> None:
        """Perform CQR computation to obtain q_hat for predictions.

        Args:
            outputs: list of dictionaries containing model outputs and labels

        """
        all_outputs = torch.cat(self.model_logits, dim=0)
        all_labels = torch.cat(self.labels, dim=0)

        # calibration quantiles assume order of outputs corresponds
        # to order of quantiles
        self.q_hat = compute_q_hat_with_cqr(all_outputs, all_labels, self.alpha)

        self.post_hoc_fitted = True

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = (
            batch[self.target_key].detach().squeeze(-1).cpu().numpy()
        )

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        if "out" in out_dict:
            del out_dict["out"]
        return out_dict

    def predict_step(self, X: Tensor) -> Dict[str, Tensor]:
        """Prediction step that produces conformalized prediction sets.

        Args:
            X: input tensor of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, "
                "please call trainer.fit(model, datamodule) first."
            )

        cqr_sets = self.forward(X)

        return {
            "pred": cqr_sets[:, 1],
            "lower_quant": cqr_sets[:, 0],
            "upper_quant": cqr_sets[:, -1],
            "out": cqr_sets,
        }

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """No optimizer needed for Conformal QR."""
        pass

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
