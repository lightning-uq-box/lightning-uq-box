"""conformalized Quantile Regression Model."""

import os
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from uq_method_box.eval_utils import compute_sample_mean_std_from_quantile

from .utils import merge_list_of_dictionaries, save_predictions_to_csv

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


class CQR(LightningModule):
    """Implements conformalized Quantile Regression.

    This should be a wrapper around any pytorch lightning model
    that conformalizes the scores and does predictions accordingly.
    """

    def __init__(
        self,
        model: LightningModule,
        quantiles: list[float],
        calibration_loader: DataLoader,
        save_dir: str,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: initialized underlying LightningModule which is the base model
                to conformalize
            quantiles: quantiles used for training and prediction
            calibration_loader: calibration data loader
            save_dir: path to directory where to save predictions
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "calibration_loader"])

        self.error_rate = 1 - max(
            self.hparams.quantiles
        )  # 1-alpha is the desired coverage

        # load model from checkpoint to conformalize it
        self.model = model

        self.cqr_fitted = False
        self.calibration_loader = calibration_loader

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

        # conformalize predictions
        cqr_sets = np.stack(
            [
                model_preds["lower_quant"] - self.q_hat,
                model_preds["mean"],
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
        outputs = [
            (self.model.predict_step(batch[0]), batch[1])
            for batch in self.calibration_loader
        ]

        # collect the quantiles into a single vector
        model_outputs = [o[0] for o in outputs]

        model_outputs = merge_list_of_dictionaries(model_outputs)
        cal_quantiles = np.stack(
            [model_outputs["lower_quant"], model_outputs["upper_quant"]], axis=-1
        )
        cal_labels = np.concatenate([o[1] for o in outputs])
        return cal_quantiles, cal_labels

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).numpy()
        return out_dict

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        save_predictions_to_csv(
            outputs, os.path.join(self.hparams.save_dir, "predictions.csv")
        )

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

        mean, std = compute_sample_mean_std_from_quantile(
            cqr_sets, self.hparams.quantiles
        )

        # can happen due to overlapping quantiles
        std[std <= 0] = 1e-6

        return {
            "mean": mean,
            "pred_uct": std,
            "lower_quant": cqr_sets[:, 0],
            "upper_quant": cqr_sets[:, -1],
            "aleatoric_uct": std,
        }
