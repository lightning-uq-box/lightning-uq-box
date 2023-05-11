"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

# TODO:
# 2) adjust loss functions such that also a two headed network output trained with nll
# works, and add mse burin-phase as in other modules
# 3) make loss function chooseable to be mse or nll like in other modules
# 7) adapt _build_model function so that
# we define a latent dimension Z neural network
# and a utility function that adds the latent dimension at a desired layer
# e.g. before last activation+linear block

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.trainers.utils import _get_input_layer_name_and_module

from uq_method_box.models.bnnlv.latent_variable_network import LatentVariableNetwork
from uq_method_box.models.bnnlv.utils import (
    get_log_f_hat,
    get_log_normalizer,
    get_log_Z_prior,
)
from uq_method_box.uq_methods.utils import _get_output_layer_name_and_module

from .bnn_vi import BNN_VI


class BNN_LV_VI(BNN_VI):
    """Bayesian Neural Network (BNN) with Latent Variables (LV) trained with Variational Inferece."""  # noqa: E501

    def __init__(
        self,
        model: nn.Module,
        latent_net: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        save_dir: str,
        num_training_points: int,
        latent_intro_layer_idx: int = 0,
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
        layer_type: str = "reparameterization",
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 1.0,
        lv_init_mu: float = 0.0,
        lv_init_std: float = 1.0,
        lv_latent_dim: int = 1,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN+LV.

        Args:
            model:
            latent_net: latent variable network
            optimizer:
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
            model,
            optimizer,
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
            layer_type,
            quantiles,
        )

        self.hparams["latent_intro_layer_idx"] = latent_intro_layer_idx
        self.hparams["lv_prior_mu"] = lv_prior_mu
        self.hparams["lv_prior_std"] = lv_prior_std
        self.hparams["lv_init_mu"] = lv_init_mu
        self.hparams["lv_init_std"] = lv_init_std
        self.hparams["lv_latent_dim"] = lv_latent_dim

        self._setup_bnn_with_vi_lv(latent_net)

    def _setup_bnn_with_vi_lv(self, latent_net: nn.Module) -> None:
        """Configure setup of BNN with VI model."""
        # TODO adjust the BNN model to adapt parameters
        # where latent variables are introduced

        # TODO or adjust input dim of latent network?
        # TODO introduce the latent variable network here

        # TODO need to check that where latent variablse are introduced
        # the following layer in the model needs to be a linear layer, right?
        _, lv_output_module = _get_output_layer_name_and_module(latent_net)
        assert lv_output_module.out_features == self.hparams.lv_latent_dim * 2, (
            "The specified latent network needs to have the same output dimension as "
            f"`lv_latent_dim` but found {lv_output_module.out_features} "
            f"and lv_latent_dim {self.hparams.lv_latent_dim}"
        )

        _, lv_input_module = _get_input_layer_name_and_module(latent_net)
        # need to find the output dimension at which latent net is introduced
        self.lv_net = LatentVariableNetwork(
            net=latent_net,
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

    def forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Forward pass BNN LV.

        Args:
            X: input data
            y: target

        Returns:
            bnn output
        """
        for idx, layer in enumerate(self.model.model):
            if idx == self.hparams.latent_intro_layer_idx:
                if y is not None:
                    z = self.lv_net(X, y)
                else:
                    z = torch.randn(X.shape[0], self.hparams.lv_latent_dim).to(
                        self.device
                    )
                X = torch.cat([X, z], -1)
            X = layer(X)

            # import pdb
            # pdb.set_trace()

        return X

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
        log_f_hat_latent_net = []

        # assume homoscedastic noise with std output_noise_scale
        output_var = torch.ones_like(y)  # * (torch.exp(self.log_aleatoric_std))

        # draw samples for all stochastic functions
        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X, y)  # pass X and y during training for lv
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # dim=1
            log_f_hat.append(get_log_f_hat(self.model))
            # latent net
            log_f_hat_latent_net.append(self.lv_net.log_f_hat_z)

        # model_preds [num_mc_samples_train, batch_size, output_dim]
        mean_out = torch.stack(model_preds, dim=0).mean(dim=0)

        energy_loss = self.energy_loss_module(
            torch.stack(pred_losses, dim=0),
            torch.stack(log_f_hat, dim=0),
            get_log_Z_prior(self.model),
            get_log_normalizer(self.model),
            self.lv_net.log_normalizer_z,  # log_normalizer_z
            torch.stack(log_f_hat_latent_net, dim=0),  # log_f_hat_z
        )
        return energy_loss, mean_out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        model_preds = []
        # TODO correctly decompose uncertainty epistemic and aleatoric with LV
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

        return {"mean": mean_out, "pred_uct": std, "epistemic_uct": std}

    # TODO optimize both bnn and lv model parameters
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
