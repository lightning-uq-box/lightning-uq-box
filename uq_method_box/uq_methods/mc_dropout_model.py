"""Mc-Dropout module."""


from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
)

from .base import BaseModel


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

        self.num_mc_samples = self.config["model"]["mc_samples"]

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        This supports

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Test Step."""
        batch = args[0]
        out_dict = self.predict_step(batch[0])
        out_dict["targets"] = batch[1].detach().squeeze(-1).numpy()
        return out_dict

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.model.train()  # activate dropout during prediction
        preds = (
            torch.stack([self.model(batch) for _ in range(self.num_mc_samples)], dim=-1)
            .detach()
            .numpy()
        )  # shape [num_samples, batch_size, num_outputs]

        mean_samples = preds[:, 0, :]

        # assume prediction with sigma
        if preds.shape[1] == 2:
            sigma_samples = preds[:, 1, :]
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
            mean = mean_samples.mean(-1)
            std = mean_samples.std(-1)
            return {"mean": mean, "pred_uct": std, "epistemic_uct": std}
