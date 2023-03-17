"""Stochastic Gradient Langevin Dynamics (SGLD) model."""

import copy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer, required

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)
from uq_method_box.uq_methods import BaseModel


class SGLD(Optimizer):
    """SGLD Optimizer.

    Adapted from
    https://github.com/izmailovpavel/understandingbdl/blob/master/swag/posteriors/sgld.py
    based on [1]: Welling, Max, and Yee W. Teh.
    "Bayesian learning via stochastic gradient Langevin dynamics."
    Proceedings of the 28th international
    conference on machine learning (ICML-11). 2011.
    """

    def __init__(
        self, params, lr=required, noise_factor=1.0, weight_decay=0.1, batch_size=256
    ):
        """Initialize new instance of SGLD Optimizer.

        Args:
            lr: learning rate
            noise_factor: noise
            weight_decay: variance of Gaussian prior on weights
            batch_size: batch size
        """
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, noise_factor=noise_factor, weight_decay=weight_decay)
        self.lr = lr
        self.batch_size = batch_size
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
        Returns: updated loss.
        """
        factor = 1 / self.batch_size
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
                    noise_factor * (2.0 * group["lr"]) ** 0.5,
                    factor * torch.randn_like(d_p),
                )

        return loss


class SGLDModel(BaseModel):
    """SGLD method for regression."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        loss_fn: str,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of SGLD model."""
        super().__init__(model_class, model_args, lr, loss_fn, save_dir)

        self.n_burnin_epochs = self.model_args.get("n_burnin_epochs")
        self.n_sgld_samples = self.model_args.get("n_sgld_samples")
        self.max_epochs = self.model_args.get("max_epochs")
        self.models: List[nn.Module] = []
        self.quantiles = quantiles
        self.weight_decay = self.model_args.get("weight_decay")
        self.lr = self.model_args.get("lr")

        assert (
            self.n_sgld_samples + self.n_burnin_epochs == self.max_epochs
        ), "The max_epochs needs to be the sum of the burnin phase and sample numbers"

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation,
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers,
            with SGLD optimizer.
        """
        optimizer = SGLD(self.model.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss and List of models
        """
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        if self.current_epoch > self.n_burnin_epochs:
            self.models.append(copy.deepcopy(self.model))

        return loss

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
        preds = (
            torch.stack([self.model(X) for self.model in self.models], dim=-1)
            .detach()
            .numpy()
        )  # shape [n_sgld_samples, batch_size, num_outputs]

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
