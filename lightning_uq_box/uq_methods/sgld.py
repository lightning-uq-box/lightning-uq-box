"""Stochastic Gradient Langevin Dynamics (SGLD) model."""
# TO DO:
# SGLD with ensembles


import os
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lightning_uq_box.uq_methods import BaseModel

from .utils import process_model_prediction


# SGLD Optimizer from Izmailov, currently in __init__.py
class SGLD(Optimizer):
    """Stochastic Gradient Langevian Dynamics Optimzer."""

    def __init__(
        self,
        params: Iterator[nn.parameter.Parameter],
        lr: float,
        noise_factor: float,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize new instance of SGLD Optimier.

        Args:
            params: model parameters
            lr: initial learning rate
            noise_factor: parameter denoting how much noise to inject in the SGD update
            weight_decay: weight decay parameter for SGLD optimizer
        """
        defaults = dict(lr=lr, noise_factor=noise_factor, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr

    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model

        Returns:
            updated loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            noise_factor = group["noise_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                p.data.add_(d_p, alpha=-group["lr"])
                p.data.add_(
                    torch.randn_like(d_p),
                    alpha=noise_factor * (2.0 * group["lr"]) ** 0.5,
                )

        return loss


class SGLDModel(BaseModel):
    """Storchastic Gradient Langevian Dynamics method for regression."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: str,
        save_dir: str,
        lr: float,
        weight_decay: float,
        noise_factor: float,
        burnin_epochs: int,
        n_sgld_samples: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of SGLD model.

        Args:
            model_class: underlying model class
            model_args: arguments to initialize *model_class*
            lr: initial learning rate
            loss_fn: choice of loss function
            save_dir: directory where to save SGLD snapshots
            weight_decay: weight decay parameter for SGLD optimizer
            noise_factor: parameter denoting how much noise to inject in the SGD update
            burnin_epochs: number of epochs to fit mse loss
            n_sgld_samples: number of sgld samples to collect
            quantiles:

        """
        super().__init__(model, None, loss_fn, save_dir)

        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.snapshot_dir = os.path.join(self.hparams.save_dir, "model_snapshots")
        os.makedirs(self.snapshot_dir)

        self.models: list[nn.Module] = []
        self.dir_list = []

        # manual optimization with SGLD optimizer
        self.automatic_optimization = False

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            SGLD optimizer and scheduler
        """
        # optimizer = SGLDAlt(params=self.parameters(), lr=self.lr)
        optimizer = SGLD(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
        )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     # "monitor": "train_loss",
            #     "interval": "epoch"
            # },
        }

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        sgld_opt = self.optimizers()
        sgld_opt.zero_grad()

        X, y = batch[self.input_key], batch[self.target_key]
        out = self.forward(X)

        def closure():
            """Closure function for optimizer."""
            sgld_opt.zero_grad()
            if self.current_epoch < self.hparams.burnin_epochs:
                loss = nn.functional.mse_loss(self.extract_mean_output(out), y)
            # after train with nll
            else:
                loss = self.loss_fn(out, y)
            loss = self.loss_fn(out, y)
            sgld_opt.zero_grad()
            self.manual_backward(loss)
            return loss

        loss = sgld_opt.step(closure=closure)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        # scheduler step every N epochs
        # scheduler = self.lr_schedulers()
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
        #     scheduler.step()

        return loss

    def on_train_epoch_end(self) -> list:
        """Save model ckpts after epoch and log training metrics."""
        # save ckpts for n_sgld_sample epochs before end (max_epochs)
        if self.current_epoch >= (
            self.trainer.max_epochs - self.hparams.n_sgld_samples
        ):
            torch.save(
                self.model.state_dict(),
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt"),
            )
            self.dir_list.append(
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt")
            )

        # log train metrics
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Predict step with SGLD, take n_sgld_sampled models, get mean and variance.

        Args:
            self: SGLD class
            batch_idx: default int=0
            dataloader_idx: default int=0

        Returns:
            output dictionary with uncertainty estimates
        """
        # create predictions from models loaded from checkpoints
        preds: list[torch.Tensor] = []
        for ckpt_path in self.dir_list:
            self.model.load_state_dict(torch.load(ckpt_path))
            preds.append(self.model(X))

        preds = torch.stack(preds, dim=-1).detach()
        # shape [batch_size, num_outputs, n_sgld_samples]

        return process_model_prediction(preds, self.hparams.quantiles)
