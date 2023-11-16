"""Deterministic Model that predicts parameters of Gaussian."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .base import DeterministicModel
from .loss_functions import NLL
from .utils import default_regression_metrics


class MVEBase(DeterministicModel):
    """Mean Variance Estimation Network Base Class.

    If you use this model in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/374138
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        burnin_epochs: int,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """Initialize a new instace of Deterministic Gaussian Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            burnin_epochs: number of burnin epochs before switiching to NLL
            lr_scheduler: learning rate scheduler

        """
        super().__init__(model, optimizer, None, None)

        self.loss_fn = NLL()

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

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


class MVERegression(MVEBase):
    """Mean Variance Estimation Model for Regression that is trained with NLL.

    If you use this model in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/374138
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        burnin_epochs: int,
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instance of Mean Variance Estimation Model for Regression.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            burnin_epochs: number of burnin epochs before switiching to NLL
            lr_scheduler: learning rate scheduler

        """
        super().__init__(model, optimizer, burnin_epochs, lr_scheduler)
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model."""
        assert out.shape[-1] <= 2, "Gaussian output."
        return out[:, 0:1]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = self.model(X)

        mean, log_sigma_2 = preds[:, 0], preds[:, 1].cpu()
        eps = torch.ones_like(log_sigma_2) * 1e-6
        std = torch.sqrt(eps + np.exp(log_sigma_2))
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
