"""Deterministic Model that predicts parameters of Gaussian."""

from typing import Any, Dict

import numpy as np
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.train_utils import NLL

from .base import BaseModel


class DeterministicGaussianModel(BaseModel):
    """Deterministic Gaussian Model that is trained with NLL."""

    def __init__(self, config: Dict[str, Any], model: nn.Module = None) -> None:
        """Initialize a new instace of Deterministic Gaussian Model."""
        super().__init__(config, model, None)

        self.criterion = NLL()

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert (
            out.shape[-1] == 2
        ), "This model should give exactly 2 outputs (mu, sigma)"
        return out[:, 0:1]

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Test Step Deterministic Gaussian Model."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        preds = self.model(X)
        mean = preds[:, 0]
        std = preds[:, 1]
        quantiles = compute_quantiles_from_std(
            mean, std, self.config["model"].get("quantiles", [0.1, 0.5, 0.9])
        )
        return {
            "mean": mean,
            "pred_uct": std,
            "aleatoric_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
