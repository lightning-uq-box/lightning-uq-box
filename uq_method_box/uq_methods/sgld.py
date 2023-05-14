"""Stochastic Gradient Langevin Dynamics (SGLD) model."""

import os
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import trange

from .base import BaseModel
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
        loss_fn: nn.Module,
        train_loader: DataLoader,
        save_dir: str,
        lr: float,
        weight_decay: float,
        noise_factor: float,
        burnin_epochs: int,
        max_epochs: int,
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

        self.snapshot_dir = os.path.join(self.hparams.save_dir, "model_snapshots")
        os.makedirs(self.snapshot_dir)

        self.hparams["burnin_epochs"] = burnin_epochs
        self.hparams["max_epochs"] = max_epochs
        self.hparams["n_sgld_samples"] = n_sgld_samples
        self.hparams["quantiles"] = quantiles
        self.hparams["weight_decay"] = weight_decay
        self.hparams["noise_factor"] = noise_factor
        self.hparams["quantiles"] = quantiles
        self.hparams["lr"] = lr
        self.hparams["weight_decay"] = weight_decay
        self.hparams["noise_factor"] = noise_factor

        self.train_loader = train_loader
        self.model_ckpt_list: list[str] = []

        # manual optimization with SGLD optimizer
        self.automatic_optimization = False

        self.sgld_fitted = False

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Not intended."""
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Not intended."""
        pass

    def on_test_start(self) -> None:
        """Fit the SGLD regime.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        if not self.sgld_fitted:
            sgld_opt = SGLD(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                noise_factor=self.hparams.noise_factor,
            )

            with torch.inference_mode(False):
                bar = trange(self.hparams.max_epochs)

                for current_epoch in bar:
                    for X, y in self.train_loader:
                        sgld_opt.zero_grad()
                        out = self.model(X)
                        loss = self.loss_fn(out, y)
                        loss.backward()
                        sgld_opt.step()

                        # save sgld snapshot
                        if current_epoch >= (
                            self.hparams.max_epochs - self.hparams.n_sgld_samples
                        ):
                            path = os.path.join(
                                self.snapshot_dir, f"{self.current_epoch}_model.ckpt"
                            )
                            torch.save(self.model.state_dict(), path)
                            self.model_ckpt_list.append(path)

            self.sgld_fitted = True

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
        if not self.sgld_fitted:
            self.on_test_start()
        # create predictions from models loaded from checkpoints
        preds: list[torch.Tensor] = []
        for ckpt_path in self.model_ckpt_list:
            self.model.load_state_dict(torch.load(ckpt_path))
            preds.append(self.model(X))

        preds = torch.stack(preds, dim=-1).detach().numpy()
        # shape [batch_size, num_outputs, n_sgld_samples]

        return process_model_prediction(preds, self.hparams.quantiles)

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            SGLD optimizer and scheduler
        """
        pass
