# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

import os
from typing import Any

import einops
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.models.bnn_layers.bnn_utils import convert_deterministic_to_bnn
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
    save_regression_predictions,
)


class BNN_VI_Base(DeterministicModel):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        bayesian_layer_type: str = "reparameterization",
        stochastic_module_names: list[str | int] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instace of BNN VI.

        Args:
            model: pytorch model that will be converted into a BNN
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
            bayesian_layer_type: `flipout` or `reparameterization`
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.
        """
        super().__init__(model, None, False, optimizer, lr_scheduler)
        assert n_mc_samples_train > 0, "Need to sample at least once during training."
        assert n_mc_samples_test > 0, "Need to sample at least once during testing."

        # update hparams
        self.hparams["weight_decay"] = 0.0
        self.save_hyperparameters(
            ignore=["model", "optimizer", "lr_scheduler", "latent_net"]
        )

        self.stochastic_module_names = map_stochastic_modules(
            self.model, stochastic_module_names
        )

        self.freeze_backbone = freeze_backbone
        self._setup_bnn_with_vi()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        pass

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams["prior_mu"],
            "prior_sigma": self.hparams["prior_sigma"],
            "posterior_mu_init": self.hparams["posterior_mu_init"],
            "posterior_rho_init": self.hparams["posterior_rho_init"],
            "layer_type": self.hparams["bayesian_layer_type"],
        }

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        convert_deterministic_to_bnn(
            self.model, self._define_bnn_args(), self.stochastic_module_names
        )

        # TODO how to best configure this parameter
        # why do we use homoscedastic noise?
        self.log_aleatoric_std = nn.Parameter(
            torch.tensor([-2.5 for _ in range(1)], device=self.device)
        )

        # only call model after it has been setup
        self.freeze_model()

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass BNN+LI.

        Args:
            X: input data

        Returns:
            bnn output
        """
        return self.model(X)

    def on_fit_start(self) -> None:
        """Before fitting compute number of training points."""
        self.num_training_points = len(
            self.trainer.datamodule.train_dataloader().dataset
        )
        self.energy_loss_module = EnergyAlphaDivergence(
            N=self.num_training_points, alpha=self.hparams["alpha"]
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the data loader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("train_loss", energy_loss, batch_size=X.shape[0])  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the data loader

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]
        energy_loss, mean_output = self.compute_energy_loss(X, y)

        self.log("val_loss", energy_loss, batch_size=X.shape[0])  # logging to Logger
        self.val_metrics(mean_output, y)

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
        self, named_params, weight_decay: float, skip_list: list[str] = ("mu", "rho")
    ):
        """Exclude non VI parameters from weight_decay optimization.

        Args:
            named_params: named parameters of the model
            weight_decay: weight decay factor
            skip_list: list of strings that if found in parameter name
                excludes the parameter from weight decay

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
        # import inspect
        # def get_function_args_defaults(func):
        #     signature = inspect.signature(func)
        #     args = []
        #     defaults = {}
        #     for name, param in signature.parameters.items():
        #         if param.default is not inspect.Parameter.empty:
        #             defaults[name] = param.default
        #         args.append(name)
        #     return args, defaults
        # args, defaults = get_function_args_defaults(self.optimizer)
        # optimizer_args = getattr(self.optimizer, "keywords")
        # wd = optimizer_args.get("weight_decay", 0.0)
        # TODO this does not work with lightning CLI correctly yet
        # self.optimizer is not a partial function anymore that can be accessed with
        # keywords using default weight decay for now

        params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=0.01)

        optimizer = self.optimizer(params)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class BNN_VI_Regression(BNN_VI_Base):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        bayesian_layer_type: str = "reparameterization",
        stochastic_module_names: list[int] | list[str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instace of BNN VI Regression.

        Args:
            model: pytorch model that will be converted into a BNN
            optimizer: optimizer used for training
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
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            bayesian_layer_type: reparameterization layer type,
                "reparametrization" or "flipout"
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.

        """
        super().__init__(
            model,
            n_mc_samples_train,
            n_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            bayesian_layer_type,
            stochastic_module_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )
        self.save_hyperparameters(ignore=["model", "optimizer", "lr_scheduler"])

        # need individual nlls of a gaussian, as we first do logsumexp over samples
        # cannot sum over batch size first as logsumexp is non-linear
        # TODO: do we support training with aleatoric output noise?
        self.nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging mean_out: mean output
                over samples, dim [n_mc_samples_train, output_dim]
        """
        model_preds: list[Tensor] = []
        pred_losses: list[Tensor] = []
        log_f_hat: list[Tensor] = []

        # assume homoscedastic noise with std output_noise_scale
        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2

        # draw samples for all stochastic functions
        for i in range(self.hparams["n_mc_samples_train"]):
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

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],  # type: ignore[override]
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        del outputs["samples"]
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        # output from forward: [n_samples, batch_size, outputs]
        with torch.no_grad():
            model_preds = torch.stack(
                [self.forward(X) for _ in range(self.hparams["n_mc_samples_test"])],
                dim=0,
            ).cpu()
        # model_preds [batch_size, output_dim]
        mean_out = model_preds.mean(dim=0)

        # how can this happen that there is so little sample diversity
        # there should be at least a little numerical difference?
        std_epistemic = model_preds.std(dim=0).squeeze()
        std_epistemic[std_epistemic <= 0] = 1e-6
        std_aleatoric = std_epistemic * 0.0 + torch.exp(self.log_aleatoric_std).detach()

        std = torch.sqrt(std_epistemic**2 + std_aleatoric**2)

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
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        bayesian_layer_type: str = "reparameterization",
        stochastic_module_names: list[str | int] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instace of BNN VI Batched.

        Args:
            model: pytorch model that will be converted into a BNN
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
            bayesian_layer_type: reparameterization layer type,
                "reparametrization" or "flipout"
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            freeze_backbone: whether to freeze the backbone
            lr_scheduler: learning rate scheduler
            optimizer: optimizer used for training

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive.
            AssertionError: if ``n_mc_samples_test`` is not positive.
        """
        super().__init__(
            model,
            n_mc_samples_train,
            n_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            bayesian_layer_type,
            stochastic_module_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

        self.save_hyperparameters(ignore=["model", "optimizer", "lr_scheduler"])

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams["prior_mu"],
            "prior_sigma": self.hparams["prior_sigma"],
            "posterior_mu_init": self.hparams["posterior_mu_init"],
            "posterior_rho_init": self.hparams["posterior_rho_init"],
            "layer_type": self.hparams["bayesian_layer_type"],
            "batched_samples": True,
            "max_n_samples": max(
                self.hparams["n_mc_samples_train"], self.hparams["n_mc_samples_test"]
            ),
        }

    def forward(self, X: Tensor, n_samples: int) -> Tensor:
        """Forward pass BNN+LI.

        Args:
            X: input data
            n_samples: number of samples to compute

        Returns:
            bnn output of shape [num_samples, batch_size, num_outputs]
        """
        batched_sample_X = einops.repeat(X, "b f -> s b f", s=n_samples)
        return self.model(batched_sample_X)

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> tuple[Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging mean_out: mean output
                over samples, dim [n_mc_samples_train, output_dim]
        """
        out = self.forward(
            X, n_samples=self.hparams["n_mc_samples_train"]
        )  # [num_samples, batch_size, output_dim]
        y = torch.tile(y[None, ...], (self.hparams["n_mc_samples_train"], 1, 1))

        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2
        # BUGS here in log_f_hat should be shape [n_samples]

        # batched sampling is implemented for a max amount of samples
        # however, self.hparams["n_mc_samples_train"] might be smaller
        # thus pick those number of sapmles from log_f_hat
        energy_loss = self.energy_loss_module(
            self.nll_loss(out, y, output_var),
            get_log_f_hat([self.model])[: self.hparams["n_mc_samples_train"]],
            get_log_Z_prior([self.model]),
            get_log_normalizer([self.model]),
            log_normalizer_z=torch.zeros(1).to(self.device),  # log_normalizer_z
            log_f_hat_z=torch.zeros(1).to(self.device),  # log_f_hat_z
        )
        return energy_loss, out.detach().mean(dim=0)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            model_preds = self.forward(X, self.hparams.n_mc_samples_test)

        mean_out = model_preds.mean(dim=0)

        std_epistemic = model_preds.std(dim=0).squeeze()
        std_epistemic[std_epistemic <= 0] = 1e-6
        std_aleatoric = std_epistemic * 0.0 + torch.exp(self.log_aleatoric_std).detach()
        std = torch.sqrt(std_epistemic**2 + std_aleatoric**2)

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
