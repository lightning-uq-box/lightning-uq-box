"""Implement Quantile Regression Model."""

from typing import Any, Dict

import numpy as np
import torch.nn as nn
from uq_regression_box.eval_utils import compute_sample_mean_std_from_quantile
from uq_regression_box.train_utils import QuantileLoss

from .base_model import BaseModel

# TODO if we want to conformalize these scores
# model.fit() # quantile regression model
# model = CQR(model)
# model.predict() # with conformalized scores


class QuantileRegressionModel(BaseModel):
    """Quantile Regression Model Wrapper."""

    def __init__(self, config: Dict[str, Any], model: nn.Module = None) -> None:
        """Initialize a new instance of Quantile Regression Model."""
        super().__init__(config, model, None)

        self.quantiles = config["model"]["quantiles"]
        self.criterion = QuantileLoss(quantiles=self.quantiles)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step with Quantile Regression.

        Args:
            batch:

        Returns:
            predicted uncertainties
        """
        out = self.model(batch).detach().numpy()  # [batch_size, len(self.quantiles)]
        median = out[:, self.quantiles.index(0.5)]
        mean, std = compute_sample_mean_std_from_quantile(out, self.quantiles)

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
