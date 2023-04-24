"""Implement a Deep Ensemble Model for prediction."""

import os
from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .utils import save_predictions_to_csv


class DeepEnsembleModel(LightningModule):
    """Base Class for different Ensemble Models."""

    def __init__(
        self,
        ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
        save_dir: str,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            save_dir: path to directory where to store prediction
            quantiles: quantile values to compute for prediction
        """
        super().__init__()
        # make hparams accessible
        self.save_hyperparameters()

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        out: list[torch.Tensor] = []
        for model_config in self.hparams.ensemble_members:
            loaded_model = model_config["model_class"].load_from_checkpoint(
                model_config["ckpt_path"]
            )
            out.append(loaded_model(X))
        return torch.stack(out, dim=-1)

    def test_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Any:
        """Compute test step for deep ensemble and log test metrics.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            dictionary of uncertainty outputs
        """
        X, y = batch
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

    def generate_ensemble_predictions(self, X: Tensor) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            the ensemble predictions
        """
        return self.forward(X)  # [batch_size, num_outputs, num_ensemble_members]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X).cpu().numpy()

        mean_samples = preds[:, 0, :]

        # assume nll prediction with sigma
        if preds.shape[1] == 2:
            log_sigma_2_samples = preds[:, 1, :]
            eps = np.ones_like(log_sigma_2_samples) * 1e-6
            sigma_samples = np.sqrt(eps + np.exp(log_sigma_2_samples))
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
        # assume mse prediction
        else:
            mean = mean_samples.mean(-1)
            std = mean_samples.std(-1)
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)

            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": std,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
