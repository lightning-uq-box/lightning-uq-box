"""Conformalized Quantile Regression Model."""

from typing import Any, List, Tuple

import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from uq_regression_box.eval_utils import compute_sample_mean_std_from_quantile

# TODO add quantile outputs to all models so they can be conformalized
# with the CQR wrapper


def compute_q_hat_with_cqr(
    cal_preds: np.ndarray, cal_labels: np.ndarray, error_rate: float
) -> float:
    """Compute q_hat which is the adjustment factor for quantiles.

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
        cal_scores, np.ceil((n + 1) * (1 - error_rate)) / n, interpolation="higher"
    )

    return q_hat


# should inherit from baseclass
class CQR(LightningModule):
    """Implements Conformalized Quantile Regression.

    This should be a wrapper around any pytorch lightning model
    that conformalizes the scores and does predictions accordingly.
    """

    def __init__(
        self, model, quantiles: List[float], calibration_loader: DataLoader
    ) -> None:
        """Initialize a new instance of CQR."""
        super().__init__()
        self.score_model = model
        self.quantiles = quantiles
        self.error_rate = 1 - max(self.quantiles)  # 1-alpha is the desired coverage

        self.cqr_fitted = False
        self.calibration_loader = calibration_loader

    # TODO maybe also something before on_test_start() like Laplace
    def on_test_start(self) -> None:
        """Before testing phase, compute q_hat."""
        # need to do one pass over the calibration set
        # so that should be passed to the model wrapper to gather
        # cal_preds and cal_labels
        if not self.cqr_fitted:
            cal_preds, cal_labels = self.compute_calibration_scores()
            self.q_hat = compute_q_hat_with_cqr(cal_preds, cal_labels, self.error_rate)
            print(self.q_hat)
            self.cqr_fitted = True

    def compute_calibration_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute calibration scores."""
        out = [(self.score_model(X), y) for X, y in self.calibration_loader]
        cal_preds = np.concatenate([o[0] for o in out])
        cal_labels = np.concatenate([o[1] for o in out])
        return cal_preds, cal_labels

    def train_step(self):
        """Train step."""
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Prediction step that produces conformalized prediction sets."""
        if not self.cqr_fitted:
            self.on_test_start()
        model_preds = self.score_model.forward(batch)
        cqr_sets = np.stack(
            [
                model_preds[:, 0] - self.q_hat,
                model_preds[:, self.quantiles.index(0.5)],
                model_preds[:, -1] + self.q_hat,
            ],
            axis=1,
        )
        mean, std = compute_sample_mean_std_from_quantile(cqr_sets, self.quantiles)
        return {
            "mean": mean,
            "median": cqr_sets[:, 1],
            "pred_uct": std,
            "lower_quant": cqr_sets[:, 0],
            "upper_quant": cqr_sets[:, -1],
            "aleatoric_uct": std,
        }
