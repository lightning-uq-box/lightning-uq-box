"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torchgeo.trainers.utils import _get_input_layer_name_and_module

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.models import LatentVariableNetwork
from uq_method_box.models.bnnlv.linear_layer import LinearReparameterization
from uq_method_box.uq_methods.utils import _get_output_layer_name_and_module

from .bnn_vi import BNN_VI


class BNN_LV_VI(BNN_VI):
    """Bayesian Neural Network (BNN) with Latent Variables (LV) trained with Variational Inferece."""  # noqa: E501

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        latent_net: nn.Module,
        lr: float,
        save_dir: str,
        num_training_points: int,
        latent_intro_layer_idx: int = 0,
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
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 1.0,
        lv_init_mu: float = 0.0,
        lv_init_std: float = 1.0,
        lv_latent_dim: int = 1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN+LV.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            latent_intro_layer_idx: at which layer index to introduce the lv
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
            lv_prior_mu: prior mean for latent variable network
            lv_prior_std: prior std for latent variable network
            lv_init_mu: initial mean for latent variable network
            lv_init_std: initial std for latent variable network
            lv_latent_dim: number of latent dimension
            quantiles: quantiles to compute

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

        self.latent_net = latent_net
        self.hparams["latent_intro_layer_idx"] = latent_intro_layer_idx
        self.hparams["lv_prior_mu"] = lv_prior_mu
        self.hparams["lv_prior_std"] = lv_prior_std
        self.hparams["lv_init_mu"] = lv_init_mu
        self.hparams["lv_init_std"] = lv_init_std
        self.hparams["lv_latent_dim"] = lv_latent_dim

        self._setup_bnn_with_vi_lv()

    def _setup_bnn_with_vi_lv(self) -> None:
        """Configure setup of BNN with VI model."""
        # TODO adjust the BNN model to adapt parameters
        # where latent variables are introduced

        # TODO or adjust input dim of latent network?
        # TODO introduce the latent variable network here

        # TODO need to check that where latent variablse are introduced
        # the following layer in the model needs to be a linear layer, right?
        _, lv_output_module = _get_output_layer_name_and_module(self.latent_net)
        assert lv_output_module.out_features == self.hparams.lv_latent_dim, (
            "The specified latent network needs to have the same output dimension as "
            f"`lv_latent_dim` but found {lv_output_module.out_features} "
            f"and lv_latent_dim {self.hparams.lv_latent_dim}"
        )

        _, lv_input_module = _get_input_layer_name_and_module(self.latent_net)
        # need to find the output dimension at which latent net is introduced
        self.lv_net = LatentVariableNetwork(
            net=self.latent_net,
            num_training_points=self.hparams.num_training_points,
            lv_prior_mu=self.hparams.lv_prior_mu,
            lv_prior_std=self.hparams.lv_prior_std,
            lv_init_mu=self.hparams.lv_init_mu,
            lv_init_std=self.hparams.lv_init_std,
            lv_latent_dim=self.hparams.lv_latent_dim,
            n_samples=self.hparams.num_mc_samples_train,
        )

        # assert that output of latent variable network is equal to latent dim
        # helper function to get the last output dimension

    def forward(self, X: Tensor, latent_idx: None, n_samples: int) -> Tensor:
        """Forward pass BNN+VI.

        Args:
            X: input data
            latent_idx: latent variable indices
            n_samples: number of samples to draw

        Returns:
            bnn output
        """
        # TODO find elegant way to handle this reliably
        for idx, layer in enumerate(self.model.model):
            if idx == self.hparams.latent_intro_layer_idx:
                # use known latent index during training
                if latent_idx is not None:
                    z = self.lv_net(X, latent_idx, n_samples=n_samples)
                # during training and testing we don't have
                # latent index so generate random prior
                else:
                    z = torch.randn(
                        n_samples, X.shape[0], self.hparams.lv_latent_dim
                    ).to(self.device)

                # adjust X for next module
                # tile x to [S,N,D]
                if len(X.shape) == 2:
                    X = X[None, :, :].repeat(z.shape[0], 1, 1)
                X = torch.cat([X, z], -1)

            # stochastic and non-stochastic layers
            # TODO introduce latent variables at desired layer
            if isinstance(layer, LinearReparameterization):
                X = layer(X, n_samples=n_samples)
            else:
                X = layer(X)
        return X

    def compute_loss(self, X: Tensor, y: Tensor, latent_idx: Tensor) -> tuple[Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
        """
        out = self.forward(
            X, latent_idx, n_samples=self.hparams.num_mc_samples_train
        )  # [num_samples, batch_size, output_dim]

        # BNN loss terms
        (
            log_Z_prior,
            log_f_hat,
            log_normalizer,
        ) = self.collect_loss_terms_over_bnn_layers()

        # Latent Variable Network Loss terms
        log_normalizer_z = self.lv_net.log_normalizer_z
        log_f_hat_z = self.lv_net.log_f_hat_z

        log_likelihood = Normal(out.transpose(0, 1), torch.exp(self.log_aleatoric_std))
        # TODO once we introduce the latent variable network, compute log_normalizer_z and log_f_hat_z # noqa: E501
        energy_loss = self.energy_loss_module(
            y,
            log_likelihood,
            log_f_hat,
            log_Z_prior,
            log_normalizer,
            log_normalizer_z,
            log_f_hat_z,
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
                self.forward(
                    X, latent_idx=None, n_samples=self.hparams.num_mc_samples_test
                )
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
