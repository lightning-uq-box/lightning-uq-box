"""Implement Quantile Regression Model."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lightning_uq_box.eval_utils import compute_sample_mean_std_from_quantile

from .base import BaseModel
from .loss_functions import HuberQLoss, QuantileLoss


class QuantileRegressionModel(BaseModel):
    """Quantile Regression Model Wrapper."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        loss_fn: nn.Module = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        save_dir: str = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Quantile Regression Model.

        Args:
            model:
            optimizer:
            save_dir:
        """
        assert all(i < 1 for i in quantiles), "Quantiles should be less than 1."
        assert all(i > 0 for i in quantiles), "Quantiles should be greater than 0."
        super().__init__(model, optimizer, loss_fn, lr_scheduler, save_dir)

        self.save_hyperparameters(ignore=["model"])
        self.median_index = self.hparams.quantiles.index(0.5)

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
    ) -> dict[str, np.ndarray]:
        """Predict step with Quantile Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            predicted uncertainties
        """
        with torch.no_grad():
            out = self.model(X)  # [batch_size, len(self.quantiles)]
            np_out = out.cpu().numpy()

        median = self.extract_mean_output(out)
        mean, std = compute_sample_mean_std_from_quantile(
            np_out, self.hparams.quantiles
        )

        # can happen due to overlapping quantiles
        std[std <= 0] = 1e-2

        return {
            "pred": median,
            "pred_uct": std,
            "lower_quant": np_out[:, 0],
            "upper_quant": np_out[:, -1],
            "aleatoric_uct": std,
        }
