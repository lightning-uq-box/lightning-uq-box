"""Bayesian Neural Networks with Variational Inference."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std

from .base import BaseModel


class BayesianNeuralNetwork_VI(BaseModel):
    """Bayesian Neural Network (BNN) with Variational Inference (VI)."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        save_dir: str,
        num_training_points: int,
        beta_elbo: float = 1.0,
        num_mc_samples_train: int = 30,
        num_mc_samples_test: int = 30,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bayesian_layer_type: str = "Reparameterization",
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
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `Flipout` or `Reparameterization`
        """
        super().__init__(model_class, model_args, lr, None, save_dir)
        self._setup_bnn_with_vi()

        # update hyperparameters
        self.hparams["num_mc_samples_train"] = num_mc_samples_train
        self.hparams["num_mc_samples_test"] = num_mc_samples_test
        self.hparams["quantiles"] = quantiles
        self.hparams["weight_decay"] = 1e-5
        self.hparams["beta_elbo"] = beta_elbo

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
        dnn_to_bnn(self.model, self.bnn_args)

        self.nll_loss = nn.GaussianNLLLoss(reduction="mean")

        # beta factor elbo
        self.beta_lambda = lambda epochs: self.hparams["beta_elbo"]

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass BNN+VI.

        Args:
            X: input data

        Returns:
            bnn output
        """
        return self.model(X)

    def compute_elbo_loss(self, X: Tensor, y: Tensor) -> Tuple[Tensor]:
        """Compute the ELBO loss.

        Args:
            X: input data
            y: target

        Returns:
            negative elbo loss and mean model output for logging
        """
        model_preds = []
        pred_losses = torch.zeros(self.hparams.num_mc_samples_train)
        kls = torch.zeros(self.hparams.num_mc_samples_train)

        output_var = torch.ones_like(y) * (1**2)

        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            model_preds.append(pred.detach())
            # compute prediction loss with nll and track over samples
            pred_losses[i] = self.nll_loss(pred, y, output_var)
            # gather and track kl losses over mc_samples
            kls[i] = get_kl_loss(self.model)

        mean_pred = torch.cat(model_preds, dim=-1).mean(-1, keepdim=True)
        mean_pred_nll_loss = torch.mean(pred_losses)
        mean_kl = torch.mean(kls)

        beta = self.beta_lambda(self.current_epoch)

        negative_beta_elbo = (
            mean_pred_nll_loss + (beta / self.hparams.num_training_points) * mean_kl
        )
        return negative_beta_elbo, mean_pred

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]
        elbo_loss, mean_output = self.compute_elbo_loss(X, y)

        self.log("train_loss", elbo_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return elbo_loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        elbo_loss, mean_output = self.compute_elbo_loss(X, y)

        self.log("val_loss", elbo_loss)  # logging to Logger
        self.train_metrics(mean_output, y)

        return elbo_loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        preds = (
            torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples_test)], dim=-1
            )
            .detach()
            .cpu()
            .numpy()
        )  # shape [num_samples, batch_size, num_outputs]

        mean = preds.mean(-1).squeeze(-1)
        std = preds.std(-1).squeeze(-1)
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
