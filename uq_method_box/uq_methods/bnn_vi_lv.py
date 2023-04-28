"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.models.bnnlv.linear_layer import LinearReparameterization

from .bnn_vi import BNN_VI


class BNN_LV_VI(BNN_VI):
    """Bayesian Neural Network (BNN) with Latent Variables (LV) trained with Variational Inferece."""  # noqa: E501

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        save_dir: str,
        num_training_points: int,
        num_stochastic_modules: int = 1,
        beta_elbo: float = 1,
        num_mc_samples_train: int = 25,
        num_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        quantiles: List[float] = ...,
    ) -> None:
        """Initialize a new instace of BNN+LV.

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
            lr,
            save_dir,
            num_training_points,
            num_stochastic_modules,
            beta_elbo,
            num_mc_samples_train,
            num_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            quantiles,
        )

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of BNN with VI model."""
        super()._setup_bnn_with_vi()

        # TODO introduce the latent variable network here

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
            # TODO introduce latent variables at desired layer
            if isinstance(layer, LinearReparameterization):
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
            if isinstance(layer, LinearReparameterization):
                log_Z_prior_terms.append(layer.calc_log_Z_prior())
                log_f_hat_terms.append(layer.log_f_hat)
                log_normalizer_terms.append(layer.log_normalizer)
            else:
                continue

        log_Z_prior = torch.stack(log_Z_prior_terms).sum(0)  # 0 shape
        log_f_hat = torch.stack(log_f_hat_terms).sum(0)  # num_samples
        log_normalizer = torch.stack(log_normalizer_terms).sum(0)  # 0 shape

        log_likelihood = Normal(out.transpose(0, 1), torch.exp(self.log_aleatoric_std))
        # TODO once we introduce the latent variable network, compute log_normalizer_z and log_f_hat_z # noqa: E501
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

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        # TODO correctly decompose epistemic and aleatoric uncertainty
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

    # TODO optimize both bnn and lv model parameters
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
