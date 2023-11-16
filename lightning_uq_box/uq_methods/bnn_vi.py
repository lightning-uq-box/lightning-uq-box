"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

import os
from typing import Any, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_quantiles_from_std
from lightning_uq_box.models.bnn_layers.utils import dnn_to_bnn_some
from lightning_uq_box.models.bnnlv.utils import (
    get_log_f_hat,
    get_log_normalizer,
    get_log_Z_prior,
)

from .base import DeterministicModel
from .loss_functions import EnergyAlphaDivergence
from .utils import (
    default_regression_metrics,
    map_stochastic_modules,
    save_predictions_to_csv,
)


class BNN_VI_Base(DeterministicModel):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
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
        lr_scheduler: type[LRScheduler] = None,
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
        super().__init__(model, optimizer, None, lr_scheduler)

        assert n_mc_samples_train > 0, "Need to sample at least once during training."
        assert n_mc_samples_test > 0, "Need to sample at least once during testing."

        # update hparams
        self.hparams.weight_decay = 0.0
        self.save_hyperparameters(ignore=["model", "latent_net"])

        self.part_stoch_module_names = map_stochastic_modules(
            self.model, part_stoch_module_names
        )

        self._setup_bnn_with_vi()

        self.pred_file_name = "predictions.csv"

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        pass

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
        dnn_to_bnn_some(
            self.model, self._define_bnn_args(), self.part_stoch_module_names
        )

        # need individual nlls of a gaussian, as we first do logsumexp over samples
        # cannot sum over batch size first as logsumexp is non-linear
        # TODO: do we support training with aleatoric output noise?
        self.nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)

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

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("train_loss", energy_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]
        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("val_loss", energy_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

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
        optimizer_args = getattr(self.optimizer, "keywords")
        wd = optimizer_args.get("weight_decay", 0.0)
        params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=wd)

        optimizer = self.optimizer(params=params)
        return optimizer


class BNN_VI_Regression(BNN_VI_Base):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        num_training_points: int,
        part_stoch_module_names: Optional[Union[list[int], list[str]]] = None,
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        layer_type: str = "reparameterization",
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instace of BNN VI Regression.

        Args:
            model: pytorch model that will be converted into a BNN
            optimizer: optimizer used for training
            num_training_points: number of data points contained in the training dataset
            part_stoch_module_names: list of module names or indices that should be converted
                to variational layers
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
            layer_type: Bayesian layer_type type, "reparametrization" or "flipout"
            lr_scheduler: learning rate scheduler

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.

        """
        super().__init__(
            model,
            optimizer,
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
            lr_scheduler,
        )
        self.save_hyperparameters(ignore=["model"])

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> None:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
            mean_out: mean output over samples, dim [batch_size, output_dim]
        """
        model_preds: list[Tensor] = []
        pred_losses: list[Tensor] = []
        log_f_hat: list[Tensor] = []

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
            torch.concat(log_f_hat, dim=0),
            get_log_Z_prior([self.model]),
            get_log_normalizer([self.model]),
            log_normalizer_z=torch.zeros(1).to(self.device),  # log_normalizer_z
            log_f_hat_z=torch.zeros(1).to(self.device),  # log_f_hat_z
        )

        return energy_loss, mean_out.detach()

    # def on_test_batch_end(
    #     self,
    #     outputs: dict[str, np.ndarray],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx=0,
    # ):
    #     """Test batch end save predictions for regression."""
    #     if self.hparams.save_dir:
    #         outputs = {key: val for key, val in outputs.items() if key != "samples"}
    #         save_predictions_to_csv(
    #             outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
    #         )

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
        model_preds = torch.stack(model_preds, dim=0).detach().cpu()
        mean_out = model_preds.mean(dim=0).squeeze()

        # how can this happen that there is so little sample diversity
        # there should be at least a little numerical difference?
        std_epistemic = model_preds.std(dim=0).squeeze()
        std_epistemic[std_epistemic <= 0] = 1e-6
        std_aleatoric = (
            std_epistemic * 0.0 + torch.exp(self.log_aleatoric_std).detach().cpu()
        )

        std = np.sqrt(std_epistemic**2 + std_aleatoric**2)

        return {
            "pred": mean_out,
            "pred_uct": std,
            "epistemic_uct": std_epistemic,
            "aleatoric_uct": std_aleatoric,
            "samples": model_preds,
        }


class BNN_VI_BatchedRegression(BNN_VI_Regression):
    """Batched sampling version of BNN_VI.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
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
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new instace of BNN VI Batched.

        Args:
            model: pytorch model that will be converted into a BNN
            optimizer: optimizer used for training
            num_training_points: number of data points contained in the training dataset
            part_stoch_module_names: list of module names or indices that should be converted
                to variational layers
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
            layer_type: Bayesian layer_type type, "reparametrization" or "flipout"
            lr_scheduler: learning rate scheduler

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.
        """
        super().__init__(
            model,
            optimizer,
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
            lr_scheduler,
        )

        self.save_hyperparameters(ignore=["model"])

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
            bnn output of shape [num_samples, batch_size, num_outputs]
        """
        batched_sample_X = einops.repeat(X, "b f -> s b f", s=n_samples)
        return self.model(batched_sample_X)

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
        )  # [num_samples, batch_size, output_dim]
        y = torch.tile(y[None, ...], (self.hparams.n_mc_samples_train, 1, 1))

        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2
        # BUGS here in log_f_hat should be shape [n_samples]

        energy_loss = self.energy_loss_module(
            self.nll_loss(out, y, output_var),
            get_log_f_hat([self.model]),
            get_log_Z_prior([self.model]),
            get_log_normalizer([self.model]),
            log_normalizer_z=torch.zeros(1).to(self.device),  # log_normalizer_z
            log_f_hat_z=torch.zeros(1).to(self.device),  # log_f_hat_z
        )
        return energy_loss, out.detach().mean(dim=0)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        preds = []
        with torch.no_grad():
            for _ in range(
                int(self.hparams.n_mc_samples_test / self.hparams.n_mc_samples_train)
            ):
                preds.append(self.forward(X).cpu())

        model_preds = torch.cat(preds, dim=0)
        mean_out = model_preds.mean(dim=0).squeeze()

        std_epistemic = model_preds.std(dim=0).squeeze()
        std_epistemic[std_epistemic <= 0] = 1e-6
        std_aleatoric = (
            std_epistemic * 0.0 + torch.exp(self.log_aleatoric_std).detach().cpu()
        )
        std = np.sqrt(std_epistemic**2 + std_aleatoric**2)

        return {
            "pred": mean_out,
            "pred_uct": std,
            "epistemic_uct": std_epistemic,
            "aleatoric_uct": std_aleatoric,
            "samples": model_preds,
        }

    def freeze_layers(self, n_samples: int) -> None:
        """Freeze BNN Layers to fix the stochasticity over forward passes.

        Args:
            n_samples: number of samples used in frozen layers
        """
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.freeze_layer(n_samples)
