"""Deterministic Model that predicts parameters of Gaussian."""

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.train_utils import NLL

from .base import BaseModel


class DeterministicGaussianModel(BaseModel):
    """Deterministic Gaussian Model that is trained with NLL."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        loss_fn: str,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model."""
        super().__init__(model_class, model_args, lr, loss_fn, save_dir)

        self.criterion = NLL()
        self.quantiles = quantiles

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

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = self.model(X)
        mean = preds[:, 0]
        std = preds[:, 1]
        quantiles = compute_quantiles_from_std(mean, std, self.quantiles)
        return {
            "mean": mean,
            "pred_uct": std,
            "aleatoric_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
