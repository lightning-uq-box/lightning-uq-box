"""Mc-Dropout module."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseModel
from .utils import process_model_prediction


class MCDropoutModel(BaseModel):
    """MC-Dropout Model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        num_mc_samples: int,
        loss_fn: nn.Module,
        burnin_epochs: int,
        save_dir: str,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model_class:
            model_args:
            num_mc_samples: number of MC samples during prediction
        """
        super().__init__(model, optimizer, loss_fn, save_dir)
        self.save_hyperparameters(ignore=["model"])

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        This supports

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
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
        out = self.forward(batch["inputs"])

        if self.current_epoch < self.hparams.burnin_epochs:
            loss = nn.functional.mse_loss(
                self.extract_mean_output(out), batch["targets"]
            )
        else:
            loss = self.loss_fn(out, batch["targets"])

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), batch["targets"])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test Step."""
        out_dict = self.predict_step(batch["inputs"])
        out_dict["targets"] = batch["targets"].detach().squeeze(-1).cpu().numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.model.train()  # activate dropout during prediction
        with torch.no_grad():
            preds = (
                torch.stack(
                    [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
                )
                .cpu()
                .numpy()
            )  # shape [batch_size, num_outputs, num_samples]

        return process_model_prediction(preds, self.hparams.quantiles)
