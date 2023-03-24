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
        optimizer: type[torch.optim.Optimizer],
        optimizer_args: Dict[str, Any],
        loss_fn: str,
        burnin_epochs: int,
        max_epochs: int,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model."""
        super().__init__(
            model_class, model_args, optimizer, optimizer_args, loss_fn, save_dir
        )

        self.criterion = NLL()
        self.quantiles = quantiles
        self.burnin_epochs = burnin_epochs
        self.max_epochs = max_epochs

        assert (
            self.burnin_epochs <= self.max_epochs
        ), "The max_epochs needs to be larger than the burnin phase."

        assert (
            self.n_outputs == 2
        ), f"Model output should be 2D but found {self.n_outputs} dimensions."

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert (
            out.shape[-1] == 2
        ), "This model should give exactly 2 outputs (mu, log_sigma_2)"
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
            loss = nn.functional.mse_loss(self.extract_mean_output(out), y)
        else:
            loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        mean, log_sigma_2 = preds[:, 0], preds[:, 1]
        eps = np.ones_like(log_sigma_2) * 1e-6
        std = np.sqrt(eps + np.exp(log_sigma_2))
        quantiles = compute_quantiles_from_std(mean, std, self.quantiles)
        return {
            "mean": mean,
            "pred_uct": std,
            "aleatoric_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
