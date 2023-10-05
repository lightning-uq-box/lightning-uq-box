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
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model_class:
            model_args:
            num_mc_samples: number of MC samples during prediction
        """
        super().__init__(model, optimizer, loss_fn, lr_scheduler, save_dir)
        self.save_hyperparameters(ignore=["model", "loss_fn"])

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
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """

        def activate_dropout(model) -> None:
            """Activate only the droput layers, model could also have batch norm layers.

            Args:
                model: nn.Module to activate dropout layers
            """
            for layer in model.children():
                if isinstance(layer, nn.Dropout):
                    layer.train()
                elif isinstance(layer, nn.Module):
                    activate_dropout(layer)

        activate_dropout(self.model)  # activate dropout layers
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_model_prediction(preds, self.hparams.quantiles)
