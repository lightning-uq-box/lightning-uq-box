"""Mc-Dropout module."""


from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from uq_regression_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
)

from .base_model import BaseModel


class MCDropoutModel(BaseModel):
    """MC-Dropout Model."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module = None,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """Initialize a new instance of MCDropoutModel."""
        super().__init__(config, model, criterion)

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions

        """
        self.train()
        preds = (
            torch.stack(
                [self.model(batch) for _ in range(self.config["model"]["mc-samples"])],
                dim=-1,
            )
            .detach()
            .numpy()
        )  # shape [num_samples, batch_size, num_outputs]

        mean_samples = preds[:, 0, :]
        sigma_samples = preds[:, 1, :]

        # assume prediction with sigma
        if preds.shape[1] == 2:
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
            }
        # assume mse prediction
        else:
            mean = preds.mean(0)
            std = preds.std(0)

            return {"mean": mean, "pred_uct": std}
