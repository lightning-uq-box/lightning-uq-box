"""Stochastic Gradient Langevin Dynamics (SGLD) model."""
# TO DO:
# SGLD with ensembles
# load params from checkpoints
#
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer, required

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)
from uq_method_box.uq_methods import BaseModel


class SGLD(Optimizer):
    """SGLD Optimzer."""

    def __init__(
        self, params, lr: float, noise_factor: float = 0.7, weight_decay: float = 0
    ):
        """Initialize new instance of SGLD Optimier."""
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, noise_factor=noise_factor, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.params = params
        self.lr = lr

    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
        Returns: updated loss.
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
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group["lr"], d_p)
                p.data.add_(
                    noise_factor * (2.0 * group["lr"]) ** 0.5, torch.randn_like(d_p)
                )

        return loss


class SGLDModel(BaseModel):
    """SGLD method for regression."""

    def __init__(
        self,
        model_class: Union[List[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        loss_fn: str,
        save_dir: str,
        max_epochs: int,
        weight_decay: float,
        n_burnin_epochs: int,
        n_sgld_samples: int,
        restart_cosine: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of SGLD model."""
        super().__init__(model_class, model_args, lr, loss_fn, save_dir)

        # makes self.hparams accesible
        self.save_hyperparameters()

        self.snapshot_dir = os.path.join(self.hparams.save_dir, "model_snapshots")
        os.makedirs(self.snapshot_dir)

        self.automatic_optimization = False
        self.n_burnin_epochs = n_burnin_epochs
        self.n_sgld_samples = n_sgld_samples
        self.max_epochs = max_epochs
        self.models: List[nn.Module] = []
        self.quantiles = quantiles
        self.weight_decay = weight_decay
        self.lr = lr
        self.restart_cosine = restart_cosine
        self.dir_list = []

        assert (
            self.n_sgld_samples + self.n_burnin_epochs == self.max_epochs
        ), "The max_epochs needs to be the sum of the burnin phase and sample numbers"

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation,
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers,
            with SGLD optimizer and cosine lr scheduler,
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        """
        optimizer = SGLD(params=self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.restart_cosine,
            T_mult=1,
            eta_min=0,
            last_epoch=-1,
            verbose=False,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    # {"optimizer": optimizer}
    #
    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss and List of models
        """
        sch = self.lr_schedulers()
        sch.step()

        opt = self.optimizers()
        opt.zero_grad()
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def on_train_epoch_end(self) -> List:
        """Save model checkpoints after train epochs."""
        if self.current_epoch > self.n_burnin_epochs:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt"),
            )
            self.dir_list.append(
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt")
            )

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
    ) -> Dict[str, np.ndarray]:
        """Predict step with SGLD, take n_sgld_sampled models, get mean and variance.

        Args:
            self: SGLD class
            batch_idx: default int=0
            dataloader_idx: default int=0

        Returns:
            "mean": sgld_mean, mean prediction over models
            "pred_uct": sgld_std, predictive uncertainty
            "epistemic_uct": sgld_epistemic, epistemic uncertainty
            "aleatoric_uct": sgld_aleatoric, aleatoric uncertainty, averaged over models
            "quantiles": sgld_quantiles, quantiles assuming output is Gaussian
        """
        # create predictions from models loaded from checkpoints
        preds: List[torch.Tensor] = []
        for ckpt_path in self.dir_list:
            self.model.load_state_dict(torch.load(ckpt_path))
            preds.append(self.model(X))

        preds = torch.stack(preds, dim=-1).detach().numpy()
        # shape [batch_size, n_sgld_samples, num_outputs]

        # Prediction gives two outputs, due to NLL loss
        mean_samples = preds[:, 0, :]

        # assume prediction with sigma
        if preds.shape[1] == 2:
            sigma_samples = preds[:, 1, :]
            sgld_mean = mean_samples.mean(-1)
            sgld_std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            sgld_aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            sgld_epistemic = compute_epistemic_uncertainty(mean_samples)
            sgld_quantiles = compute_quantiles_from_std(
                sgld_mean, sgld_std, self.quantiles
            )
            return {
                "mean": sgld_mean,
                "pred_uct": sgld_std,
                "epistemic_uct": sgld_epistemic,
                "aleatoric_uct": sgld_aleatoric,
                "lower_quant": sgld_quantiles[:, 0],
                "upper_quant": sgld_quantiles[:, -1],
            }
        # assume mse prediction
        else:
            sgld_mean = mean_samples.mean(-1)
            sgld_std = mean_samples.std(-1)
            sgld_quantiles = compute_quantiles_from_std(
                sgld_mean, sgld_std, self.quantiles
            )
            return {
                "mean": sgld_mean,
                "pred_uct": sgld_std,
                "epistemic_uct": sgld_std,
                "lower_quant": sgld_quantiles[:, 0],
                "upper_quant": sgld_quantiles[:, -1],
            }
