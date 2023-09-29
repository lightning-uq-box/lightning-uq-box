"""Deterministic Model that predicts parameters of Gaussian."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .base import BaseModel
from .loss_functions import NLL


class DeterministicGaussianModel(BaseModel):
    """Deterministic Gaussian Model that is trained with NLL."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        burnin_epochs: int,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        save_dir: str = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model."""
        super().__init__(model, optimizer, None, None, save_dir)

        self.loss_fn = NLL()
        self.save_hyperparameters(ignore=["model"])

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

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        out = self.forward(batch[self.input_key])

        if self.current_epoch < self.hparams.burnin_epochs:
            loss = nn.functional.mse_loss(
                self.extract_mean_output(out), batch[self.target_key]
            )
        else:
            loss = self.loss_fn(out, batch[self.target_key])

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), batch[self.target_key])

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = self.model(X)

        mean, log_sigma_2 = preds[:, 0], preds[:, 1].cpu().numpy()
        eps = np.ones_like(log_sigma_2) * 1e-6
        std = np.sqrt(eps + np.exp(log_sigma_2))
        quantiles = compute_quantiles_from_std(
            mean.cpu().numpy(), std, self.hparams.quantiles
        )
        return {
            "pred": mean,
            "pred_uct": std,
            "aleatoric_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
            "out": preds,
        }
