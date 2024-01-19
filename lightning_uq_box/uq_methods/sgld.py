# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Stochastic Gradient Langevin Dynamics (SGLD) model."""
# TO DO:
# SGLD with ensembles


import os
from collections.abc import Iterator
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lightning_uq_box.uq_methods import DeterministicModel

from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    process_regression_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


# SGLD Optimizer from Izmailov, currently in __init__.py
class SGLD(Optimizer):
    """Stochastic Gradient Langevian Dynamics Optimzer.

    If you use this optimizer in your research, please cite the following paper:

    * https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
    """

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


class SGLDBase(DeterministicModel):
    """Storchastic Gradient Langevian Dynamics method for regression.

    If you use this model in your research, please cite the following paper:

    * https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float,
        weight_decay: float,
        noise_factor: float,
        n_sgld_samples: int,
    ) -> None:
        """Initialize a new instance of SGLD model.

        Args:
            model: pytorch model
            loss_fn: choice of loss function
            lr: initial learning rate for SGLD optimizer
            weight_decay: weight decay parameter for SGLD optimizer
            noise_factor: parameter denoting how much noise to inject in the SGD update
            burnin_epochs: number of epochs to fit mse loss
            n_sgld_samples: number of sgld samples to collect
        """
        super().__init__(model, loss_fn, None, None)

        self.save_hyperparameters(ignore=["model", "loss_fn"])

        self.models: list[nn.Module] = []
        self.dir_list = []

        # manual optimization with SGLD optimizer
        self.automatic_optimization = False

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            SGLD optimizer and scheduler
        """
        optimizer = SGLD(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
        )
        return {"optimizer": optimizer}

    def on_train_start(self) -> None:
        """On training start."""
        self.snapshot_dir = os.path.join(
            self.trainer.default_root_dir, "model_snapshots"
        )
        os.makedirs(self.snapshot_dir)

    def on_train_epoch_end(self) -> None:
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


class SGLDRegression(SGLDBase):
    """Stochastic Gradient Langevin Dynamics method for regression."""

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float,
        weight_decay: float,
        noise_factor: float,
        burnin_epochs: int,
        n_sgld_samples: int,
    ) -> None:
        """Initialize a new instance of SGLD model.

        Args:
            model: pytorch model
            loss_fn: choice of loss function
            lr: initial learning rate for SGLD optimizer
            weight_decay: weight decay parameter for SGLD optimizer
            noise_factor: parameter denoting how much noise to inject in the SGD update
            burnin_epochs: number of epochs to fit mse loss
            n_sgld_samples: number of sgld samples to collect
        """
        super().__init__(model, loss_fn, lr, weight_decay, noise_factor, n_sgld_samples)
        self.burnin_epochs = burnin_epochs

    def setup_task(self) -> None:
        """Set up task specific metrics."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

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
        sgld_opt = self.optimizers()
        sgld_opt.zero_grad()

        X, y = batch[self.input_key], batch[self.target_key]
        out = self.forward(X)

        def closure():
            """Closure function for optimizer."""
            sgld_opt.zero_grad()
            if self.current_epoch < self.hparams.burnin_epochs:
                loss = nn.functional.mse_loss(self.adapt_output_for_metrics(out), y)
            # after train with nll
            else:
                loss = self.loss_fn(out, y)
            sgld_opt.zero_grad()
            self.manual_backward(loss)
            return loss

        loss = sgld_opt.step(closure=closure)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.adapt_output_for_metrics(out), y)

        # return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
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

        return process_regression_prediction(preds)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class SGLDClassification(SGLDBase):
    """Stochastic Gradient Langevin Dynamics method for classification."""

    valid_tasks = ["multiclass", "binary", "multilabel"]
    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float,
        weight_decay: float,
        noise_factor: float,
        task: str = "multiclass",
        n_sgld_samples: int = 20,
    ) -> None:
        """Initialize a new instance of SGLD model.

        Args:
            model: pytorch model to train with SGLD
            loss_fn: choice of loss function
            lr: initial learning rate
            weight_decay: weight decay parameter for SGLD optimizer
            noise_factor: parameter denoting how much noise to inject in the SGD update
            task: classification task, one of ["multiclass", "binary", "multilabel"]
            n_sgld_samples: number of sgld samples to collect

        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = _get_num_outputs(model)
        super().__init__(model, loss_fn, lr, weight_decay, noise_factor, n_sgld_samples)

    def setup_task(self) -> None:
        """Set up task specific metrics."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out

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
            loss = self.loss_fn(out, y)
            sgld_opt.zero_grad()
            self.manual_backward(loss)
            return loss

        loss = sgld_opt.step(closure=closure)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.adapt_output_for_metrics(out), y)

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with SGLD, take n_sgld_sampled models, get mean and variance.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

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

        return process_classification_prediction(preds)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )
