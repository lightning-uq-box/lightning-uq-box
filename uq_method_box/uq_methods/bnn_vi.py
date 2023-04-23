"""Bayesian Neural Networks with Variational Inference."""

# TODO:
# change dnn_to_bnn function such that only some layers are made stochastic done!
# adjust loss functions such that also a two headed network output trained with nll
# works, and add mse burin-phase as in other modules
# make loss function chooseable to be mse or nll like in other modules
# probably only use this implementation and remove bayes_by_backprop.py

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.models.bnnlv.linear_layer import LinearFlipoutLayer
from uq_method_box.models.bnnlv.utils import linear_dnn_to_bnn
from uq_method_box.train_utils.loss_functions import EnergyAlphaDivergence

from .base import BaseModel

# TODO separate this model class into BNN and BNN+LV and have BNN+LV inherit from BNN probably # noqa: E501


class BNN_VI(BaseModel):
    """Bayesian Neural Network (BNN) with Variational Inference (VI)."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        save_dir: str,
        num_training_points: int,
        num_stochastic_modules: int = 1,
        beta_elbo: float = 1.0,
        num_mc_samples_train: int = 25,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Model instance.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            beta_elbo: beta factor for negative elbo loss computation
            num_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive.
            AssertionError: if ``num_mc_samples_test`` is not positive.
        """
        super().__init__(
            model_class,
            model_args,
            optimizer=torch.optim.Adam,
            optimizer_args={"lr": lr},
            loss_fn=None,
            save_dir=save_dir,
        )

        assert num_mc_samples_train > 0, "Need to sample at least once during training."
        assert num_mc_samples_test > 0, "Need to sample at least once during testing."

        self.save_hyperparameters()

        self._setup_bnn_with_vi()

        # update hyperparameters
        self.hparams["num_mc_samples_train"] = num_mc_samples_train
        self.hparams["num_mc_samples_test"] = num_mc_samples_test
        self.hparams["quantiles"] = quantiles
        self.hparams["weight_decay"] = 1e-5
        self.hparams["beta_elbo"] = beta_elbo
        self.hparams["output_noise_scale"] = output_noise_scale

        self.hparams["prior_mu"] = prior_mu
        self.hparams["prior_sigma"] = prior_sigma
        self.hparams["posterior_mu_init"] = posterior_mu_init
        self.hparams["posterior_rho_init"] = posterior_rho_init
        self.hparams["num_training_points"] = num_training_points
        self.hparams["num_stochastic_modules"] = num_stochastic_modules
        self.hparams["alpha"] = alpha

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        self.bnn_args = {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
        }
        # convert deterministic model to BNN
        linear_dnn_to_bnn(self.model, self.bnn_args)

        self.energy_loss_module = EnergyAlphaDivergence(
            N=self.hparams.num_training_points, alpha=self.hparams.alpha
        )

        # TODO how to best configure this parameter
        self.log_aleatoric_std = nn.Parameter(
            torch.tensor([1.0 for _ in range(1)], device=self.device)
        )

    def forward(self, X: Tensor, n_samples: int) -> Tensor:
        """Forward pass BNN+VI.

        Args:
            X: input data
            n_samples: number of samples to draw

        Returns:
            bnn output
        """
        # TODO find elegant way to handle this reliably
        for layer in self.model.model:
            # stochastic and non-stochastic layers
            if isinstance(layer, LinearFlipoutLayer):
                X = layer(X, n_samples=n_samples)
            else:
                X = layer(X)
        return X

    def compute_loss(self, X: Tensor, y: Tensor) -> tuple[Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
        """
        out = self.forward(
            X, n_samples=self.hparams.num_mc_samples_train
        )  # [num_samples, batch_size, output_dim]

        # compute loss terms over layer
        log_Z_prior_terms = []
        log_f_hat_terms = []
        log_normalizer_terms = []
        for layer in self.model.model:
            if isinstance(layer, LinearFlipoutLayer):
                log_Z_prior_terms.append(layer.calc_log_Z_prior())
                log_f_hat_terms.append(layer.log_f_hat)
                log_normalizer_terms.append(layer.log_normalizer)
            else:
                continue

        log_Z_prior = torch.stack(log_Z_prior_terms).sum(0)  # 0 shape
        log_f_hat = torch.stack(log_f_hat_terms).sum(0)  # num_samples
        log_normalizer = torch.stack(log_normalizer_terms).sum(0)  # 0 shape

        log_likelihood = Normal(out.transpose(0, 1), torch.exp(self.log_aleatoric_std))

        energy_loss = self.energy_loss_module(
            y,
            log_likelihood,
            log_f_hat,
            log_Z_prior,
            log_normalizer,
            0.0,  # log_normalizer_z
            0.0,  # log_f_hat_z
        )
        return energy_loss, out.mean(dim=0)

    # *(beta / self.hparams.num_training_points)
    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]

        energy_loss, mean_output = self.compute_loss(X, y)

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
        energy_loss, mean_output = self.compute_elbo_loss(X, y)

        self.log("val_loss", energy_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return energy_loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        # output from forward: [num_samples, batch_size, outputs]
        with torch.no_grad():
            preds = (
                self.forward(X, n_samples=self.hparams.num_mc_samples_test)
                .cpu()
                .squeeze(-1)
            )

        # currently only single output, might want to support NLL output as well
        mean = preds.mean(0)
        std = preds.std(0)
        quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
        return {
            "mean": mean,
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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.weight_decay
        )

        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer
