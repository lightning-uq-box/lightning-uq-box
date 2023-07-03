"""Bayesian Neural Networks with Variational Inference."""

# TODO:
# adapt to new config file scheme

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from torch import Tensor
from omegaconf import OmegaConf, ListConfig

from .base import BaseModel
from .loss_functions import NLL
from .utils import dnn_to_bnn_some, process_model_prediction, map_stochastic_modules


class BNN_VI_ELBO(BaseModel):
    """Bayes By Backprop Model with Variational Inference (VI)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        loss_fn: nn.Module,
        burnin_epochs: int,
        num_training_points: int,
        part_stoch_module_names: int = None,
        beta: float = 100,
        num_mc_samples_train: int = 10,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        bayesian_layer_type: str = "Reparameterization",
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        save_dir: str = None,
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
            bayesian_layer_type: `Flipout` or `Reparameterization`

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        super().__init__(model, optimizer, loss_fn, lr_scheduler, save_dir)

        assert num_mc_samples_train > 0, "Need to sample at least once during training."
        assert num_mc_samples_test > 0, "Need to sample at least once during testing."

        self.part_stoch_module_names = map_stochastic_modules(
            self.model, part_stoch_module_names
        )

        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self._setup_bnn_with_vi()

        # update hyperparameters
        self.hparams["weight_decay"] = 1e-5

        # hyperparameter depending on network size
        self.beta = beta

        self.burnin_epochs = burnin_epochs
        self.criterion = NLL()
        self.lr_scheduler = lr_scheduler


    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        self.bnn_args = {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "type": self.hparams.bayesian_layer_type,
            "moped_enable": False,
        }
        # convert deterministic model to BNN
        dnn_to_bnn_some(
            self.model,
            self.bnn_args,
            part_stoch_module_names=self.part_stoch_module_names,
        )

        # TODO we currently have self.criterion as NLL and GaussianNLL
        # should remove one and could it just be the self.loss_fn of the base module?
        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass BNN+VI.

        Args:
            X: input data

        Returns:
            bnn output
        """
        return self.model(X)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        This supports

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def compute_elbo_loss(self, X: Tensor, y: Tensor, mse=True) -> tuple[Tensor]:
        """Compute the ELBO loss with mse/nll.

        Args:
            X: input data
            y: target

        Returns:
            negative elbo loss and mean model output [batch_size]
            for logging
        """
        model_preds = []
        pred_losses = torch.zeros(self.hparams.num_mc_samples_train)

        # assume homoscedastic noise with std output_noise_scale
        output_var = torch.ones_like(y) * (self.hparams.output_noise_scale**2)

        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            model_preds.append(self.extract_mean_output(pred).detach())
            # compute prediction loss with nll and track over samples
            if mse:
                # compute mse loss with output noise scale, is like mse
                pred_losses[i] = self.nll_loss(
                    self.extract_mean_output(pred), y, output_var
                )
            else:
                # after burnin compute nll with log_sigma
                pred_losses[i] = self.criterion(pred, y)

        mean_pred = torch.cat(model_preds, dim=-1).mean(-1, keepdim=True)
        # dimension [batch_size]

        mean_pred_nll_loss = torch.mean(pred_losses)
        # shape 0, mean over batch_size, this is "the S factor":)
        # need to potentially multiply by full training set size

        mean_kl = get_kl_loss(self.model)

        negative_beta_elbo = (
            self.hparams.num_training_points * mean_pred_nll_loss + self.beta * mean_kl
        )

        return negative_beta_elbo, mean_pred

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = batch["inputs"], batch["targets"]

        if self.current_epoch < self.burnin_epochs:
            mse = True
        elif self.current_epoch >= self.burnin_epochs and self.num_outputs == 1:
            mse = True
        else:
            mse = False

        elbo_loss, mean_output = self.compute_elbo_loss(X, y, mse)

        self.log("train_loss", elbo_loss)  # logging to Logger
        if batch["inputs"].shape[0] > 1:
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
        X, y = batch["inputs"], batch["targets"]
        if self.num_outputs == 1:
            mse = True
        else:
            mse = False

        elbo_loss, mean_output = self.compute_elbo_loss(X, y, mse)

        self.log("val_loss", elbo_loss)  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.val_metrics(mean_output, y)

        return elbo_loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step."""
        out_dict = self.predict_step(batch["inputs"])
        out_dict["targets"] = batch["targets"].detach().squeeze(-1).cpu().numpy()

        if self.num_outputs == 1:
            mse = True
        else:
            mse = False

        # TODO this is inefficient, since we are doing repeated forward passes just for loss computation
        # therefore adapt predict and test step
        self.log("test_loss", self.compute_elbo_loss(batch["inputs"], batch["targets"], mse)[0])  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch["targets"])

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            preds = (
                torch.stack(
                    [self.model(X) for _ in range(self.hparams.num_mc_samples_test)], dim=-1
                )
            )  # shape [batch_size, num_outputs, num_samples]

        return process_model_prediction(preds, self.hparams.quantiles)

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
