"""conformalized Quantile Regression Model."""

import numbers
import os
from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor

from lightning_uq_box.eval_utils import compute_sample_mean_std_from_quantile

from .base import BaseModule
from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    merge_list_of_dictionaries,
    save_predictions_to_csv,
)

# TODO add quantile outputs to all models so they can be conformalized
# with the CQR wrapper


def compute_q_hat_with_cqr(
    cal_preds: np.ndarray, cal_labels: np.ndarray, error_rate: float
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
    cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)

    # Get the score quantile
    q_hat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - error_rate)) / n, method="higher"
    )

    return q_hat


class ConformalPredictionBase(BaseModule):
    """Base class for conformal prediction methods."""

    def __init__(self, model: LightningModule) -> None:
        """Initialize a new Base Model.

        Args:
            model: initialized underlying LightningModule which is the base model
                to conformalize
        """
        super().__init__()
        self.model = model

        self.cqr_fitted = False

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.model.model)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension to model
        """
        return _get_num_outputs(self.model.model)

    def forward(self, X: Tensor, **kwargs: Any) -> np.ndarray:
        """Conformalized Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the model
        """
        if not self.cqr_fitted:
            self.on_test_start()

        # predict with underlying model
        with torch.no_grad():
            model_preds: dict[str, np.ndarray] = self.model.predict_step(X)

        cqr_sets = self.conformalize_predictions(model_preds)

        return cqr_sets


class CQR(ConformalPredictionBase):
    """Implements Conformalized Quantile Regression."""

    def __init__(
        self, model: LightningModule, quantiles: list[float] = [0.1, 0.5, 0.9]
    ) -> None:
        super().__init__(model)

        self.save_hyperparameters(ignore=["model"])

        self.error_rate = 1 - max(
            self.hparams.quantiles
        )  # 1-alpha is the desired coverage

    def conformalize_predictions(
        self, model_preds: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Conformalize predictions.

        Args:
            model_preds: dictionary of model predictions

        Returns:
            conformalized prediction sets
        """
        # conformalize predictions
        cqr_sets = np.stack(
            [
                model_preds["lower_quant"] - self.q_hat,
                model_preds["pred"].squeeze(-1).cpu().numpy(),
                model_preds["upper_quant"] + self.q_hat,
            ],
            axis=1,
        )
        return cqr_sets

    def on_test_start(self) -> None:
        """Before testing phase, compute q_hat."""
        # need to do one pass over the calibration set
        # so that should be passed to the model wrapper to gather
        # cal_preds and cal_labels
        if not self.cqr_fitted:
            cal_quantiles, cal_labels = self.compute_calibration_scores()
            self.q_hat = compute_q_hat_with_cqr(
                cal_quantiles, cal_labels, self.error_rate
            )
            self.cqr_fitted = True

    def compute_calibration_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration scores."""
        # model predict steps return a dictionary that contains quantiles
        calibration_loader = self.trainer.datamodule.val_dataloader()

        model_outputs: list[dict[str, Tensor]] = []
        cal_labels: list[np.ndarray] = []
        for batch in calibration_loader:
            aug_batch = self.trainer.datamodule.on_after_batch_transfer(
                batch, dataloader_idx=0
            )
            model_outputs.append(
                self.model.predict_step(aug_batch[self.input_key].to(self.device))
            )
            cal_labels.append(aug_batch[self.target_key].numpy())

        cal_labels = np.concatenate(cal_labels)
        model_outputs = merge_list_of_dictionaries(model_outputs)
        cal_quantiles = np.stack(
            [model_outputs["lower_quant"], model_outputs["upper_quant"]], axis=-1
        )
        return cal_quantiles, cal_labels

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step."""
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = (
            batch[self.target_key].detach().squeeze(-1).cpu().numpy()
        )

        # if batch[self.input_key].shape[0] > 1:
        #     self.test_metrics(
        #         out_dict["pred"].squeeze(), batch[self.target_key].squeeze(-1)
        #     )

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        if "out" in out_dict:
            del out_dict["out"]
        return out_dict

    # def on_test_batch_end(
    #     self,
    #     outputs: dict[str, np.ndarray],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx=0,
    # ):
    #     """Test batch end save predictions."""
    #     if self.save_dir:
    #         save_predictions_to_csv(
    #             outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
    #         )

    # def on_test_epoch_end(self):
    #     """Log epoch-level test metrics."""
    #     self.log_dict(self.test_metrics.compute())
    #     self.test_metrics.reset()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Prediction step that produces conformalized prediction sets.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.cqr_fitted:
            self.on_test_start()

        cqr_sets = self.forward(X)

        return {
            "pred": torch.from_numpy(cqr_sets[:, 1]).to(self.device),
            "lower_quant": cqr_sets[:, 0],
            "upper_quant": cqr_sets[:, -1],
            "out": torch.from_numpy(cqr_sets).to(self.device),
        }
