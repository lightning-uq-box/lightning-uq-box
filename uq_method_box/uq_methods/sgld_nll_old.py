"""Stochastic Gradient Langevin Dynamics (SGLD) model."""
# TO DO:
# SGLD with ensembles


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


# SGLD Optimizer from Izmailov, currently in __init__.py
class SGLD(Optimizer):
    """SGLD Optimzer."""

    def __init__(
        self, params, lr: float, noise_factor: float = 0.8, weight_decay: float = 0
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

                p.data.add_(d_p, alpha=-group["lr"])
                p.data.add_(
                    torch.randn_like(d_p),
                    alpha=noise_factor * (2.0 * group["lr"]) ** 0.5,
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
        burnin_epochs: int,
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
        self.burnin_epochs = burnin_epochs
        self.n_sgld_samples = n_sgld_samples
        self.max_epochs = max_epochs
        self.models: List[nn.Module] = []
        self.quantiles = quantiles
        self.weight_decay = weight_decay
        self.lr = lr
        self.restart_cosine = restart_cosine
        self.dir_list = []

        assert (
            self.n_sgld_samples + self.burnin_epochs <= self.max_epochs
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
        # lr scheduler, step after every training step,
        # could also be call on_train_epoch_end
        sch = self.lr_schedulers()
        sch.step()

        opt = self.optimizers()

        # manual optimization does not handle gradient accumulation
        # - maybe this is necessary?
        X, y = args[0]
        out = self.forward(X)

        # burnin phase for nll with mse loss
        if self.current_epoch < self.burnin_epochs:
            loss = nn.functional.mse_loss(self.extract_mean_output(out), y)
        # after train with nll
        else:
            loss = self.criterion(out, y)

        # apparently it can change convergence vastly
        # based on where you call opt.zero_grad(),
        #  this is from
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html
        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradients
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        opt.step()

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def on_train_epoch_end(self) -> List:
        """Save model ckpts after epoch and log training metrics."""
        # save ckpts for n_sgld_sample epochs before end (max_epochs)
        if self.current_epoch > (self.max_epochs - self.n_sgld_samples):
            torch.save(
                self.model.state_dict(),
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt"),
            )
            self.dir_list.append(
                os.path.join(self.snapshot_dir, f"{self.current_epoch}_model.ckpt")
            )

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
        # shape [batch_size, num_outputs, n_sgld_samples]

        # Prediction gives two outputs, due to NLL loss
        mean_samples = preds[:, 0, :]

        # assume prediction with sigma
        if preds.shape[1] == 2:
            log_sigma_2_samples = preds[:, 1, :]
            eps = np.ones_like(log_sigma_2_samples) * 1e-6
            sigma_samples = np.sqrt(eps + np.exp(log_sigma_2_samples))
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
