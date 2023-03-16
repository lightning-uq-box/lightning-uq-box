"""Implement Quantile Regression Model."""

from typing import Any, Dict, List, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import compute_sample_mean_std_from_quantile
from uq_method_box.train_utils import QuantileLoss

from .base import BaseModel


class QuantileRegressionModel(BaseModel):
    """Quantile Regression Model Wrapper."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Quantile Regression Model."""
        super().__init__(model_class, model_args, lr, "quantile", save_dir)

        self.median_index = self.hparams.quantiles.index(0.5)
        self.criterion = QuantileLoss(quantiles=self.hparams.quantiles)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean/median prediction from quantile regression model.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, self.median_index : self.median_index + 1]  # noqa: E203

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step with Quantile Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            predicted uncertainties
        """
        out = self.model(X).detach().numpy()  # [batch_size, len(self.quantiles)]
        median = out[:, self.median_index]
        mean, std = compute_sample_mean_std_from_quantile(out, self.hparams.quantiles)

        # can happen due to overlapping quantiles
        std[std <= 0] = 1e-6

        return {
            "mean": mean,
            "median": median,
            "pred_uct": std,
            "lower_quant": out[:, 0],
            "upper_quant": out[:, -1],
            "aleatoric_uct": std,
        }
