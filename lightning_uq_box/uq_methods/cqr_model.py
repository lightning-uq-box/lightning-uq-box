"""conformalized Quantile Regression Model."""

import math
import numbers
import os
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from .base import PosthocBase
from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    merge_list_of_dictionaries,
    save_predictions_to_csv,
)


def compute_q_hat_with_cqr(
    cal_preds: Tensor, cal_labels: Tensor, error_rate: float
) -> float:
    """Compute q_hat which is the adjustment factor for quantiles.

    Check trusted computation here.

    Args:
        cal_preds: calibration set predictions
        cal_labels: calibration set targets
        error_rate: desired error rate for quantile

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
        cal_scores, math.ceil((n + 1) * (1 - error_rate)) / n, interpolation="higher"
    )

    return q_hat


class ConformalQR(PosthocBase):
    """Conformalized Quantile Regression.

    If you use this model, please cite the following paper:

    * https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
    """

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

        self.error_rate = 1 - max(
            self.hparams.quantiles
        )  # 1-alpha is the desired coverage

    def adjust_model_output(self, model_output: Tensor) -> Tensor:
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

    def on_validation_start(self) -> None:
        """Before validation epoch starts, create tensors that gather model outputs and labels."""
        # TODO intitialize zero tensors for memory efficiency
        self.model_outputs = []
        self.labels = []

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Single CQR gathering step.

        Args:
            batch: batch of data
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            underlying model output and labels
        """
        self.model_outputs.append(self.model(batch[self.input_key]))
        self.labels.append(batch[self.target_key])

    def on_validation_epoch_end(self) -> None:
        """Perform CQR computation to obtain q_hat for predictions.

        Args:
            outputs: list of dictionaries containing model outputs and labels

        """
        # `outputs` is a list of dictionaries, each containing 'output' and 'label' from each validation step
        all_outputs = torch.cat(self.model_outputs, dim=0)
        all_labels = torch.cat(self.labels, dim=0)

        # calibration quantiles assume order of outputs corresponds to order of quantiles
        self.q_hat = compute_q_hat_with_cqr(all_outputs, all_labels, self.error_rate)

        self.post_hoc_fitted = True

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step."""
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

    def predict_step(self, X: Tensor) -> Any:
        """Prediction step that produces conformalized prediction sets.

        Args:
            X: input tensor of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, please call trainer.validate(model, datamodule) first."
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
