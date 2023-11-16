"""Bayesian Neural Networks with Variational Inference."""

# TODO:
# adapt to new config file scheme

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.models.bnn_layers.utils import dnn_to_bnn_some, get_kl_loss

from .base import DeterministicModel
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    map_stochastic_modules,
    process_classification_prediction,
    process_regression_prediction,
)


class BNN_VI_ELBO_Base(DeterministicModel):
    """Bayes By Backprop Base with Variational Inference (VI).

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/1505.05424
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        criterion: nn.Module,
        num_training_points: int,
        part_stoch_module_names: Optional[list[Union[int, str]]] = None,
        beta: float = 100,
        num_mc_samples_train: int = 10,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        bayesian_layer_type: str = "reparameterization",
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """Initialize a new Model instance.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            beta: beta factor for negative elbo loss computation,
                should be number of weights and biases
            num_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss. When setting num_mc_samples_train=1, this
                is just Bayes by Backprop.
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `flipout` or `reparameterization`

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        super().__init__(model, optimizer, criterion, lr_scheduler)

        assert num_mc_samples_train > 0, "Need to sample at least once during training."
        assert num_mc_samples_test > 0, "Need to sample at least once during testing."

        self.part_stoch_module_names = map_stochastic_modules(
            self.model, part_stoch_module_names
        )

        self.save_hyperparameters(ignore=["model", "criterion"])
        self._setup_bnn_with_vi()

        # update hyperparameters
        self.hparams["weight_decay"] = 1e-5

        # hyperparameter depending on network size
        self.beta = beta

        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def setup_task(self) -> None:
        pass

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        self.bnn_args = {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.bayesian_layer_type,
        }
        # convert deterministic model to BNN
        dnn_to_bnn_some(
            self.model,
            self.bnn_args,
            part_stoch_module_names=self.part_stoch_module_names,
        )

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass BNN+VI.

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

        elbo_loss, mean_output = self.compute_elbo_loss(X, y)

        self.log("train_loss", elbo_loss)  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(mean_output, y)

        return elbo_loss

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

        elbo_loss, mean_output = self.compute_elbo_loss(X, y)

        self.log("val_loss", elbo_loss)  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(mean_output, y)

        return elbo_loss

    def compute_elbo_loss(self, X: Tensor, y: Tensor) -> tuple[Tensor]:
        """Compute the ELBO loss with mse/nll.

        Args:
            X: input data
            y: target

        Returns:
            negative elbo loss and mean model output [batch_size]
            for logging
        """
        model_preds: list[Tensor] = []
        pred_losses = torch.zeros(self.hparams.num_mc_samples_train)

        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            pred_losses[i] = self.compute_task_loss(pred, y)
            model_preds.append(self.extract_mean_output(pred).detach())

        mean_pred = torch.stack(model_preds, dim=-1).mean(-1)
        # dimension [batch_size]

        mean_pred_nll_loss = torch.mean(pred_losses)
        # shape 0, mean over batch_size, this is "the S factor":)
        # need to potentially multiply by full training set size

        mean_kl = get_kl_loss(self.model)

        negative_beta_elbo = (
            self.hparams.num_training_points * mean_pred_nll_loss + self.beta * mean_kl
        )

        return negative_beta_elbo, mean_pred

    def compute_task_loss(self, X: Tensor, y: Tensor) -> Tensor:
        """Compute the loss for the respective task for a single sampling iteration.

        Args:
            X: input data
            y: target

        Returns:
            nll loss for the task
        """
        raise NotImplementedError

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
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class BNN_VI_ELBO_Regression(BNN_VI_ELBO_Base):
    """Bayes By Backprop Model with Variational Inference (VI) for Regression.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/1505.05424
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        criterion: nn.Module,
        burnin_epochs: int,
        num_training_points: int,
        part_stoch_module_names: Optional[Union[list[int], list[str]]] = None,
        beta: float = 100,
        num_mc_samples_train: int = 10,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        bayesian_layer_type: str = "reparameterization",
        lr_scheduler: type[LRScheduler] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Model instance.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            beta: beta factor for negative elbo loss computation,
                should be number of weights and biases
            num_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss. When setting num_mc_samples_train=1, this
                is just Bayes by Backprop.
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `flipout` or `reparameterization`

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            num_training_points,
            part_stoch_module_names,
            beta,
            num_mc_samples_train,
            num_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bayesian_layer_type,
            lr_scheduler,
        )

        self.save_hyperparameters(ignore=["model", "criterion"])

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def compute_task_loss(self, pred: Tensor, y: Tensor) -> Tensor:
        """Compute the loss for the respective task for a single sampling iteration.

        Args:
            X: model_prediction
            y: target

        Returns:
            nll loss for the task
        """
        if self.current_epoch < self.hparams.burnin_epochs:
            # compute mse loss with output noise scale, is like mse
            loss = torch.nn.functional.mse_loss(self.extract_mean_output(pred), y)
        else:
            # after burnin compute nll with log_sigma
            loss = self.criterion(pred, y)
        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples_test)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_regression_prediction(preds, self.hparams.quantiles)


class BNN_VI_ELBO_Classification(BNN_VI_ELBO_Base):
    """Bayes By Backprop Model with Variational Inference (VI) for Classification.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/1505.05424
    """

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        criterion: nn.Module,
        num_training_points: int,
        task: str = "multiclass",
        part_stoch_module_names: Optional[Union[list[int], list[str]]] = None,
        beta: float = 100,
        num_mc_samples_train: int = 10,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        bayesian_layer_type: str = "reparameterization",
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new Model instance.

        Args:
            num_training_points: number of data points contained in the training dataset
            beta: beta factor for negative elbo loss computation,
                should be number of weights and biases
            num_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss. When setting num_mc_samples_train=1, this
                is just Bayes by Backprop.
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `flipout` or `reparameterization`

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        assert task in self.valid_tasks
        self.task = task

        self.num_classes = _get_num_outputs(model)

        super().__init__(
            model,
            optimizer,
            criterion,
            num_training_points,
            part_stoch_module_names,
            beta,
            num_mc_samples_train,
            num_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            bayesian_layer_type,
            lr_scheduler,
        )

        self.save_hyperparameters(ignore=["model", "criterion"])

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def compute_task_loss(self, pred: Tensor, y: Tensor) -> Tensor:
        """Compute the loss for the respective task for a single sampling iteration.

        Args:
            X: model_prediction
            y: target

        Returns:
            nll loss for the task
        """
        return self.criterion(pred, y)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model output."""
        return out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples_test)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_classification_prediction(preds)
