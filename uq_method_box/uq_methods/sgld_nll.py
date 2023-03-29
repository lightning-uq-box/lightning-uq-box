"""Stochastic Gradient Langevin Dynamics (SGLD) model."""
# TO DO:
# SGLD with ensembles

import math
import os
import warnings
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)
from uq_method_box.uq_methods import BaseModel

# different learning rate schedulders
# to use from https://github.com/activatedgeek/torch-sgld


class ABAnnealingLR(_LRScheduler):
    r"""Step size scheduler for SGLD.

    Args:
        optimizer
        final_lr: final learning rate
        gamma: value in Wilson paper 0.55, choose sth (0,1)
        T_max: max_epochs

    Returns:
        lr: learning rate \epsilon_t scheduled by
            .. math:: \epsilon_t = a(b + t)^{-\gamma}
            and a and b are computed based on start and final step size.
         .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    """

    def __init__(self, optimizer, final_lr, gamma, T_max, last_epoch=-1, verbose=False):
        """Initialize new annealing lr scheduler."""
        self.final_lr = final_lr
        self.gamma = gamma
        self.T_max = T_max

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute new learning rate."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs

        new_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self.last_epoch > self.T_max:
                new_lrs.append(group["lr"])
            else:
                b = self.T_max / (
                    (base_lr / self.final_lr) * math.exp(1 / self.gamma) - 1.0
                )
                a = base_lr * b**self.gamma

                new_lr = a / (b + self.last_epoch) ** self.gamma
                new_lrs.append(new_lr)

        return new_lrs


class CosineLR(_LRScheduler):
    r"""Cyclic size scheduler for SGLD (a.k.a cSG-MCMC).

    Args:
        optimizer
        n_cycles: M is the number of cycles, K is the number of total iterations.
        n_samples: optionally samples can be taken at a certain epoch
        T_max: max_epochs
        beta: beta is the fraction of the cycle for which we do optimization.

    Returns:
        new_lrs: new learning rate
        (and more options not used currently)
        .. math::
        \alpha_k = \frac{\alpha_0}{2}
        \\left[ \\cos{\frac{\\pi\\mod{k-1, \\ceil{K/M}}}{\\ceil{K/M}}} \right]
        .. _cSG-MCMC\\:
        Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning:
          https://arxiv.org/abs/1902.03932
    """

    def __init__(
        self,
        optimizer,
        n_cycles,
        n_samples,
        T_max,
        beta=1 / 4,
        last_epoch=-1,
        verbose=False,
    ):
        """Instanciate new Cosine lr scheduler."""
        self.beta = beta
        self._cycle_len = int(math.ceil(T_max / n_cycles))
        self._last_beta = 0.0

        samples_per_cycle = n_samples // n_cycles
        self._thres = (
            (
                beta
                + torch.arange(1, samples_per_cycle + 1)
                * (1 - beta)
                / samples_per_cycle
            )
            * self._cycle_len
        ).int()

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute new lr."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs

        beta = (self.last_epoch % self._cycle_len) / self._cycle_len

        new_lrs = []
        _lr_factor = math.cos(math.pi * beta) + 1.0
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = 0.5 * base_lr * _lr_factor
            new_lrs.append(new_lr)

        self._last_beta = beta

        return new_lrs

    def get_last_beta(self):
        """Return last beta."""
        return self._last_beta

    def _get_closed_form_lr(self):
        """Compute closed form lr."""
        beta = (self.last_epoch % self._cycle_len) / self._cycle_len

        closed_form_lrs = []
        _lr_factor = math.cos(math.pi * beta) + 1.0
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            lr = 0.5 * base_lr * _lr_factor
            closed_form_lrs.append(lr)

        return closed_form_lrs

    def should_sample(self):
        """Aim for (n_samples // n_cycles) samples per cycle.

        note: Use before the next step() call to scheduler.
        """
        _t = self.last_epoch % self._cycle_len + 1
        return (_t - self._thres).abs().min() == 0


def sgld(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Tensor],
    *,
    weight_decay: float,
    lr: float,
    momentum: float,
    noise: bool,
    temperature: float,
):
    r"""Functional API for SGMCMC/SGHMC.

    .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    .. _SGHMC\: Stochastic Gradient Hamiltonian Monte Carlo:
          http://www.istc-cc.cmu.edu/publications/papers/2014/Guestrin-stochastic-gradient.pdf
    """
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p.add_(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            buf.mul_(1 - momentum).add_(d_p, alpha=-lr)
            if noise:
                eps = torch.randn_like(d_p)
                buf.add_(eps, alpha=math.sqrt(2 * lr * momentum * temperature))

            param.add_(buf)
        else:
            param.add_(d_p, alpha=-lr)

            if noise:
                eps = torch.randn_like(d_p)
                param.add_(eps, alpha=math.sqrt(2 * lr * temperature))


class SGLD(SGD):
    """SGLD/SGHMC Optimizer and assumes negative log density.

    SGHMC updates are used for non-zero momentum values. The gradient noise
    variance is assumed to be zero. Mass matrix is kept to be identity.

    The variance estimate of gradients is assumed to be zero for SGHMC.

    Args:
        *args: model parameters
        momentum: momentum for SGD, SGLD, default 0
        temperature: scaling of posterior, default 1

    Returns:
        loss: updated loss
    """

    def __init__(self, *args, momentum=0, temperature=1, **kwargs):
        """Initialize new SGLD optimizer."""
        super().__init__(*args, momentum=momentum, **kwargs)

        self.T = temperature
        if momentum != 0:
            self.reset_momentum()

    @torch.no_grad()
    def step(self, closure=None, noise=True):
        """Update loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgld(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                lr=lr,
                momentum=momentum,
                noise=noise,
                temperature=self.T,
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss

    @torch.no_grad()
    def reset_momentum(self):
        """Reset momentum."""
        for group in self.param_groups:
            momentum = group["momentum"]

            assert momentum > 0, "Must use momentum > 0 to use SGHMC."

            for p in group["params"]:
                state = self.state[p]
                state["momentum_buffer"] = torch.zeros_like(p)

        return self


# SGLD model with different optimizer and lr_scheduler
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
        # restart_cosine: int, not needed here
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
        # self.lr = lr
        # self.restart_cosine = restart_cosine
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
        optimizer = SGLD(
            params=self.model.parameters(), lr=self.hparams.lr, momentum=0.9
        )

        # chose lr scheduler below
        # scheduler =
        # ABAnnealingLR(
        # optimizer,
        # final_lr=self.hparams.lr/100,
        # gamma = 0.55,
        # T_max=self.max_epochs)
        scheduler = CosineLR(
            optimizer, n_cycles=100, n_samples=200, T_max=self.max_epochs
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

        if self.current_epoch < self.burnin_epochs:
            loss = nn.functional.mse_loss(self.extract_mean_output(out), y)
        else:
            loss = self.criterion(out, y)

        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def on_train_epoch_end(self) -> List:
        """Save model checkpoints after train epochs.

        Log epoch-level training metrics.
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

        if self.current_epoch > (self.max_epochs - self.n_sgld_samples):
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
