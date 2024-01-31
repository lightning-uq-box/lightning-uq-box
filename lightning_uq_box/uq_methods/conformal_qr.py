# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""conformalized Quantile Regression Model."""

import copy
import math
import os
from typing import Dict, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

from .base import PosthocBase
from .utils import default_regression_metrics, save_regression_predictions


def compute_q_hat_with_cqr(
    lower_quant: Tensor, upper_quant: Tensor, cal_labels: Tensor, alpha: float
) -> float:
    """Compute q_hat which is the adjustment factor for quantiles.

    Check trusted computation here.

    Args:
        lower_quant: lower quantile predictions
        upper_quant: upper quantile predictions
        cal_labels: calibration set targets
        alpha: 1 - alpha is desired error rate for quantile

    Returns:
        q_hat the computed quantile by which prediction intervals
        can be adjusted according to cqr
    """
    cal_labels = cal_labels.squeeze()
    n = cal_labels.shape[0]

    # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
    cal_scores = torch.maximum(cal_labels - upper_quant, lower_quant - cal_labels)

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
        alpha: float = 0.1,
    ) -> None:
        """Initialize a new CQR model.

        Args:
            model: underlying model to be wrapped
            quantiles: quantiles to be used for CQR
            alpha: 1 - alpha is desired error rate for quantile
        """
        super().__init__(model)

        self.save_hyperparameters(ignore=["model"])

        self.quantiles = quantiles

        assert alpha > 0 and alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha

        self.desired_coverage = 1 - self.alpha  # 1-alpha is the desired coverage

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task."""
        self.test_metrics = default_regression_metrics("test")

    def forward(self, X: Tensor) -> dict[str, Tensor]:
        """Forward pass of model.

        Args:
            X: input tensor of shape [batch_size x input_dims]

        Returns:
            model output tensor of shape [batch_size x num_outputs]
        """
        with torch.no_grad():
            if hasattr(self.model, "predict_step"):
                pred = self.model.predict_step(X)
            else:
                pred = self.model(X)

        pred = self.adjust_model_logits(pred)

        return pred

    def adjust_model_logits(
        self, model_output: Union[dict[str, Tensor], Tensor]
    ) -> dict[str, Tensor]:
        """Conformalize underlying model output.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            conformalized model predictions
        """
        if isinstance(model_output, dict):
            output_dict = copy.deepcopy(model_output)
            output_dict["lower_quant"] = model_output["lower_quant"] - self.q_hat
            output_dict["upper_quant"] = model_output["upper_quant"] + self.q_hat
        else:
            output_dict: dict[str, Tensor] = {}
            # conformalize predictions assum ordering of quantiles
            output_dict["lower_quant"] = model_output[:, 0] - self.q_hat
            output_dict["pred"] = model_output[:, 1]
            output_dict["upper_quant"] = model_output[:, -1] + self.q_hat

        return output_dict

    def on_validation_epoch_end(self) -> None:
        """Perform CQR computation to obtain q_hat for predictions.

        Args:
            outputs: list of dictionaries containing model outputs and labels

        """
        all_outputs = torch.cat(self.model_logits, dim=0)
        all_labels = torch.cat(self.labels, dim=0)

        # calibration quantiles assume order of outputs corresponds
        # to order of quantiles
        self.q_hat = compute_q_hat_with_cqr(
            all_outputs[:, 0], all_outputs[:, -1], all_labels, self.alpha
        )

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
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        self.test_metrics(out_dict["pred"], out_dict[self.target_key])

        # save metadata
        out_dict = self.add_aux_data_to_dict(out_dict, batch)

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

        return cqr_sets

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
