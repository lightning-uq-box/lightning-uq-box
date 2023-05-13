"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std

# this is almost like get kl_loss
from uq_method_box.models.bnnlv.utils import (
    dnn_to_bnnlv_some,
    get_log_f_hat,
    get_log_normalizer,
    get_log_Z_prior,
)

from .base import BaseModel
from .loss_functions import EnergyAlphaDivergence


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
        stochastic_module_names: Optional[list[Union[str, int]]] = None,
        num_mc_samples_train: int = 25,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        layer_type: str = "reparameterization",
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN+LV.

        Args:
            model:
            optimizer:
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            stochastic_module_names:
            num_mc_samples_train: number of MC samples during training when computing
                the energy loss
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            type: Bayesian layer_type type, "reparametrization" or "flipout"

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        super().__init__(model, optimizer, None, save_dir)

        assert num_mc_samples_train > 0, "Need to sample at least once during training."
        assert num_mc_samples_test > 0, "Need to sample at least once during testing."

        self.save_hyperparameters(ignore=["model"])

        self._setup_bnn_with_vi()

        # update hyperparameters
        self.hparams["num_mc_samples_train"] = num_mc_samples_train
        self.hparams["num_mc_samples_test"] = num_mc_samples_test
        self.hparams["quantiles"] = quantiles
        self.hparams["weight_decay"] = 1e-5
        self.hparams["output_noise_scale"] = output_noise_scale

        self.hparams["prior_mu"] = prior_mu
        self.hparams["prior_sigma"] = prior_sigma
        self.hparams["posterior_mu_init"] = posterior_mu_init
        self.hparams["posterior_rho_init"] = posterior_rho_init
        self.hparams["num_training_points"] = num_training_points
        self.hparams["stochastic_module_names"] = stochastic_module_names
        self.hparams["alpha"] = alpha
        self.hparams["layer_type"] = layer_type

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        self.bnn_args = {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.layer_type,
        }
        # TODO check what is num_stocha
        # convert deterministic model to BNN
        # model_modules = list(self.model.named_modules())
        # model_module_names = [module[0] for module in model_modules]

        dnn_to_bnnlv_some(
            self.model, self.bnn_args, self.hparams.stochastic_module_names
        )

        # need individual nlls of a gaussian, as we first do logsumexp over samples
        # cannot sum over batch size first as logsumexp is non-linear
        self.nll_loss = nn.GaussianNLLLoss(reduction="none")

        self.energy_loss_module = EnergyAlphaDivergence(
            N=self.hparams.num_training_points, alpha=self.hparams.alpha
        )

        # TODO how to best configure this parameter
        # why do we use homoscedastic noise?
        self.log_aleatoric_std = nn.Parameter(
            torch.tensor([-2.5 for _ in range(1)], device=self.device)
        )

    # can we add the latent variable here?
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
            dim [num_mc_samples_train, output_dim]
        """
        model_preds = []
        pred_losses = []
        log_f_hat = []

        # assume homoscedastic noise with std output_noise_scale
        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std))

        # draw samples for all stochastic functions
        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # dim=1
            log_f_hat.append(get_log_f_hat([self.model]))

        # model_preds [num_mc_samples_train, batch_size, output_dim]
        mean_out = torch.stack(model_preds, dim=0).mean(dim=0)

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
        model_preds = []

        # output from forward: [num_samples, batch_size, outputs]
        with torch.no_grad():
            # draw samples for all stochastic functions
            for i in range(self.hparams.num_mc_samples_train):
                # mean prediction
                pred = self.forward(X)
                model_preds.append(pred.detach())
                # model_preds [num_mc_samples_train, batch_size, output_dim]

        # model_preds [num_mc_samples_train, batch_size, output_dim]
        model_preds = torch.stack(model_preds, dim=0)

        mean_out = model_preds.mean(dim=0).squeeze(-1).cpu().numpy()
        std = model_preds.std(dim=0).squeeze(-1).cpu().numpy()

        # currently only single output, might want to support NLL output as well
        quantiles = compute_quantiles_from_std(mean_out, std, self.hparams.quantiles)
        return {
            "mean": mean_out,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }

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
