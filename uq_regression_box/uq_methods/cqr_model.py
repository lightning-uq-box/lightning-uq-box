"""Conformalized Quantile Regression Model."""

from typing import Any, List

import numpy as np
from pytorch_lightning import LightningModule


class CQR(LightningModule):
    """Implements Conformalized Quantile Regression."""

    def __init__(
        self,
        model,
        quantiles: List[float],
        cal_preds: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """Initialize a new instance of CQR."""
        super().__init__()
        self.score_model = model
        self.quantiles = quantiles
        self.error_rate = 1 - max(self.quantiles)  # 1-alpha is the desired coverage

        self.q_hat = self.compute_q_hat(cal_preds, cal_labels)

    def compute_q_hat(self, cal_preds: np.ndarray, cal_labels: np.ndarray) -> float:
        """Compute q_hat which is the adjustment factor for quantiles."""
        cal_labels = cal_labels.squeeze()

        n = cal_labels.shape[0]
        cal_upper = cal_preds[:, -1]
        cal_lower = cal_preds[:, 0]

        # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
        cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)

        # Get the score quantile
        q_hat = np.quantile(
            cal_scores,
            np.ceil((n + 1) * (1 - self.error_rate)) / n,
            interpolation="higher",
        )

        return q_hat

    def train_step(self):
        """Train step."""
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Prediction step that produces conformalized prediction sets."""
        model_preds = self.score_model.forward(batch)
        cqr_sets = np.stack(
            [
                model_preds[:, 0] - self.q_hat,
                model_preds[:, self.quantiles.index(0.5)],
                model_preds[:, -1] + self.q_hat,
            ],
            axis=1,
        )
        return cqr_sets
