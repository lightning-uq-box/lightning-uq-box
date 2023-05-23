"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

from typing import Any, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.models.bnnlv.utils import (
    dnn_to_bnnlv_some,
    get_log_f_hat,
    get_log_normalizer,
    get_log_Z_prior,
)

from .base import BaseModel
from .loss_functions import EnergyAlphaDivergence
from .utils import map_stochastic_modules


class BNN_VI(BaseModel):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        save_dir: str,
        num_training_points: int,
        part_stoch_module_names: Optional[list[Union[str, int]]] = None,
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        layer_type: str = "reparameterization",
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN VI.

        Args:
            model:
            optimizer:
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            part_stoch_module_names:
            n_mc_samples_train: number of MC samples during training when computing
                the energy loss
            n_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            type: Bayesian layer_type type, "reparametrization" or "flipout"

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.
        """
        super().__init__(model, optimizer, None, save_dir)

        assert n_mc_samples_train > 0, "Need to sample at least once during training."
        assert n_mc_samples_test > 0, "Need to sample at least once during testing."

        # update hparams
        self.save_hyperparameters(ignore=["model", "latent_net"])

        self.part_stoch_module_names = map_stochastic_modules(
            self.model, part_stoch_module_names
        )

        self._setup_bnn_with_vi()

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.layer_type,
        }

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        dnn_to_bnnlv_some(
            self.model, self._define_bnn_args(), self.part_stoch_module_names
        )

        # need individual nlls of a gaussian, as we first do logsumexp over samples
        # cannot sum over batch size first as logsumexp is non-linear
        # TODO: do we support training with aleatoric output noise?
        self.nll_loss = nn.GaussianNLLLoss(reduction="none")

        self.energy_loss_module = EnergyAlphaDivergence(
            N=self.hparams.num_training_points, alpha=self.hparams.alpha
        )

        # TODO how to best configure this parameter
        # why do we use homoscedastic noise?
        self.log_aleatoric_std = nn.Parameter(
            torch.tensor([-2.5 for _ in range(1)], device=self.device)
        )

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass BNN+LI.

        Args:
            X: input data

        Returns:
            bnn output
        """
        return self.model(X)

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> None:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
            mean_out: mean output over samples,
                dim [batch_size, output_dim]
        """
        model_preds = []
        pred_losses = []
        log_f_hat = []

        # assume homoscedastic noise with std output_noise_scale

        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2

        # draw samples for all stochastic functions
        for i in range(self.hparams.n_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # dim=1
            log_f_hat.append(get_log_f_hat([self.model]))

        # model_preds [batch_size, output_dim, n_mc_samples_train, ]
        mean_out = torch.stack(model_preds, dim=-1).mean(dim=-1)

        # TODO once we introduce the latent variable network, compute log_normalizer_z and log_f_hat_z # noqa: E501
        energy_loss = self.energy_loss_module(
            torch.stack(pred_losses, dim=0),
            torch.stack(log_f_hat, dim=0),
            get_log_Z_prior([self.model]),
            get_log_normalizer([self.model]),
            log_normalizer_z=torch.zeros(1).to(self.device),  # log_normalizer_z
            log_f_hat_z=torch.zeros(1).to(self.device),  # log_f_hat_z
        )

        return energy_loss, mean_out

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]

        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("train_loss", energy_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("val_loss", energy_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        # output from forward: [n_samples, batch_size, outputs]
        with torch.no_grad():
            model_preds = [
                self.forward(X) for _ in range(self.hparams.n_mc_samples_test)
            ]

        # model_preds [batch_size, output_dim]
        model_preds = torch.stack(model_preds, dim=-1)

        mean_out = model_preds.mean(dim=-1).squeeze(-1).cpu().numpy()
        std = model_preds.std(dim=-1).squeeze(-1).cpu().numpy()
        std[std <= 0] = 1e-6

        # currently only single output, might want to support NLL output as well
        quantiles = compute_quantiles_from_std(mean_out, std, self.hparams.quantiles)
        return {
            "mean": mean_out,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }

    def freeze_layers(self) -> None:
        """Freeze BNN Layers to fix the stochasticity over forward passes."""
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.freeze_layer()

    def unfreeze_layers(self) -> None:
        """Unfreeze BNN Layers to make them fully stochastic."""
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.unfreeze_layer()

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=("mu", "rho")
    ):
        """Exclude non VI parameters from weight_decay optimization.

        Args:
            named_params:
            weight_decay:
            skip_list:

        Returns:
            split parameter groups for optimization with and without
            weight_decay
        """
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.weight_decay
        )

        optimizer = self.optimizer(params=params)
        return optimizer


class BNN_VI_Batched(BNN_VI):
    """Batched sampling version of BNN_VI."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        save_dir: str,
        num_training_points: int,
        part_stoch_module_names: Optional[list[Union[str, int]]] = None,
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        layer_type: str = "reparameterization",
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN VI Batched.

        Args:
            model:
            optimizer:
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            part_stoch_module_names:
            n_mc_samples_train: number of MC samples during training when computing
                the energy loss
            n_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            type: Bayesian layer_type type, "reparametrization" or "flipout"

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.
        """
        super().__init__(
            model,
            optimizer,
            save_dir,
            num_training_points,
            part_stoch_module_names,
            n_mc_samples_train,
            n_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            layer_type,
            quantiles,
        )

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.layer_type,
            "batched_samples": True,
            "max_n_samples": max(
                self.hparams.n_mc_samples_train, self.hparams.n_mc_samples_test
            ),
        }

    def forward(self, X: Tensor, n_samples: int = 5) -> Tensor:
        """Forward pass BNN+LI.

        Args:
            X: input data
            n_samples: number of samples to compute

        Returns:
            bnn output of shape [batch_size, num_outputs, num_samples]
        """
        batched_sample_X = einops.repeat(X, "b f -> s b f", s=n_samples)
        return self.model(batched_sample_X).permute(1, 2, 0)

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> None:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
            mean_out: mean output over samples,
                dim [n_mc_samples_train, output_dim]
        """
        out = self.forward(
            X, n_samples=self.hparams.n_mc_samples_train
        )  # [batch_size, output_dim, num_samples]

        y = einops.repeat(y, "b f -> b f s", s=self.hparams.n_mc_samples_train)

        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2
        energy_loss = self.energy_loss_module(
            self.nll_loss(out, y, output_var),
            get_log_f_hat([self.model]),
            get_log_Z_prior([self.model]),
            get_log_normalizer([self.model]),
            log_normalizer_z=torch.zeros(1).to(self.device),  # log_normalizer_z
            log_f_hat_z=torch.zeros(1).to(self.device),  # log_f_hat_z
        )
        return energy_loss, out.mean(dim=-1)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            model_preds = self.forward(X, self.hparams.n_mc_samples_test)

        mean_out = model_preds.mean(dim=-1).squeeze(-1).cpu().numpy()
        std = model_preds.std(dim=-1).squeeze(-1).cpu().numpy()
        std[std <= 0] = 1e-6
        # currently only single output, might want to support NLL output as well
        quantiles = compute_quantiles_from_std(mean_out, std, self.hparams.quantiles)
        return {
            "mean": mean_out,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }

    def freeze_layers(self, n_samples: int) -> None:
        """Freeze BNN Layers to fix the stochasticity over forward passes.

        Args:
            n_samples: number of samples used in frozen layers
        """
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.freeze_layer(n_samples)
