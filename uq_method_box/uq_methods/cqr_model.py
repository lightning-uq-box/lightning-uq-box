"""conformalized Quantile Regression Model."""

import os
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.trainers.utils import _get_input_layer_name_and_module

from uq_method_box.eval_utils import compute_sample_mean_std_from_quantile
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score

from .utils import (
    _get_output_layer_name_and_module,
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


class CQR(LightningModule):
    """Implements conformalized Quantile Regression.

    This should be a wrapper around any pytorch lightning model
    that conformalizes the scores and does predictions accordingly.
    """

    def __init__(
        self,
        model: LightningModule,
        quantiles: list[float],
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
        self.save_hyperparameters(ignore=["model"])

        self.error_rate = 1 - max(
            self.hparams.quantiles
        )  # 1-alpha is the desired coverage

        # load model from checkpoint to conformalize it
        self.model = model

        self.cqr_fitted = False

        self.save_dir = save_dir

        self.pred_file_name = "predictions.csv"

        self.test_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(),
            },
            prefix="test_",
        )

    @property
    def num_inputs(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_input_layer_name_and_module(self.model.model)
        if hasattr(module, "in_features"):  # Linear Layer
            num_inputs = module.in_features
        elif hasattr(module, "in_channels"):  # Conv Layer
            num_inputs = module.in_channels
        return num_inputs

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_output_layer_name_and_module(self.model.model)
        if hasattr(module, "out_features"):  # Linear Layer
            num_outputs = module.out_features
        elif hasattr(module, "out_channels"):  # Conv Layer
            num_outputs = module.out_channels
        return num_outputs

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

        # calibration_loader.collate_fn = collate_fn_torch
        model_outputs = [
            self.model.predict_step(self.trainer.datamodule.on_after_batch_transfer(batch, dataloader_idx=0)["inputs"].to(self.device))
            for batch in calibration_loader
        ]
        model_outputs: list[dict[str, Tensor]] = []
        cal_labels: list[np.ndarray] = []
        for batch in calibration_loader:
            aug_batch = self.trainer.datamodule.on_after_batch_transfer(batch, dataloader_idx=0)
            model_outputs.append(self.model.predict_step(aug_batch["inputs"].to(self.device)))
            cal_labels.append(aug_batch["targets"].numpy())
        
        cal_labels = np.concatenate(cal_labels)
        model_outputs = merge_list_of_dictionaries(model_outputs)
        cal_quantiles = np.stack(
            [model_outputs["lower_quant"], model_outputs["upper_quant"]], axis=-1
        )
        return cal_quantiles, cal_labels

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        if self.save_dir:
            save_predictions_to_csv(
                outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
            )

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

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
            "pred": torch.from_numpy(mean).to(self.device),
            "pred_uct": std,
            "lower_quant": cqr_sets[:, 0],
            "upper_quant": cqr_sets[:, -1],
            "aleatoric_uct": std,
            "out": torch.from_numpy(cqr_sets).to(self.device)
        }
