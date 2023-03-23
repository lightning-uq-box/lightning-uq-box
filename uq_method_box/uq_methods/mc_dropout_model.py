"""Mc-Dropout module."""


from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)
from uq_method_box.train_utils import NLL

from .base import BaseModel


class MCDropoutModel(BaseModel):
    """MC-Dropout Model."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        num_mc_samples: int,
        lr: float,
        loss_fn: str,
        burnin_epochs: int,
        max_epochs: int,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model_class:
            model_args:
            num_mc_samples: number of MC samples during prediction
        """
        super().__init__(model_class, model_args, lr, loss_fn, save_dir)

        self.criterion = NLL()
        self.quantiles = quantiles
        self.num_mc_samples = num_mc_samples
        self.burnin_epochs = burnin_epochs
        self.max_epochs = max_epochs

        assert (
            self.burnin_epochs <= self.max_epochs
        ), "The max_epochs needs to be larger than the burnin phase."

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        This supports

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]
        out = self.forward(X)

        if self.current_epoch < self.burnin_epochs:
            loss = nn.funnctional.mse_loss(self.extract_mean_output(out), y)

        else:
            loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Test Step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).cpu().numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.model.train()  # activate dropout during prediction
        with torch.no_grad():
            preds = (
                torch.stack([self.model(X) for _ in range(self.num_mc_samples)], dim=-1)
                .detach()
                .numpy()
            )  # shape [num_samples, batch_size, num_outputs]

        mean_samples = preds[:, 0, :]

        # assume prediction with sigma
        if preds.shape[1] == 2:
            log_sigma_2 = preds[:, 1, :]
            eps = np.ones_like(log_sigma_2) * 1e-6
            sigma_samples = np.sqrt(eps + np.exp(log_sigma_2))
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            quantiles = compute_quantiles_from_std(mean, std, self.quantiles)
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
            quantiles = compute_quantiles_from_std(mean, std, self.quantiles)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": std,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
