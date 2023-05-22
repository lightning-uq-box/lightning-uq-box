"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

# TODO:
# 2) adjust loss functions such that also a two headed network output trained with nll
# works, and add mse burin-phase as in other modules
# 3) make loss function chooseable to be mse or nll like in other modules
# 7) adapt _build_model function so that
# we define a latent dimension Z neural network
# and a utility function that adds the latent dimension at a desired layer
# e.g. before last activation+linear block

import math
from typing import Any, Optional, Union

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
    replace_module,
    retrieve_module_init_args,
)
from uq_method_box.uq_methods.utils import _get_output_layer_name_and_module

from .bnn_vi import BNN_VI


class BNN_LV_VI(BNN_VI):
    """Bayesian Neural Network (BNN) with Latent Variables (LV) trained with Variational Inferece."""  # noqa: E501

    lv_intro_options = ["first", "last"]

    def __init__(
        self,
        model: nn.Module,
        latent_net: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        save_dir: str,
        num_training_points: int,
        stochastic_module_names: list[Union[str, int]] = 1,
        latent_variable_intro: str = "first",
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
        # lv_init_mu: float = 0.0,
        lv_init_std: float = 1.0,
        lv_latent_dim: int = 1,
        init_scaling: float = 0.01,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN+LV.

        Args:
            model:
            latent_net: latent variable network
            optimizer:
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            stochastic_module_names:
            latent_variable_intro:
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
            #lv_init_mu: initial mean for latent variable network
            lv_init_std: initial std for latent variable network
            lv_latent_dim: number of latent dimension
            quantiles: quantiles to compute

        Raises:
            AssertionError: if ``num_mc_samples_train`` is not positive
            AssertionError: if ``num_mc_samples_test`` is not positive
        """
        super().__init__(
            model,
            optimizer,
            save_dir,
            num_training_points,
            stochastic_module_names,
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

        assert (
            latent_variable_intro in self.lv_intro_options
        ), f"Only one of {self.lv_intro_options} is possible, but found {latent_variable_intro}."  # noqa: E501
        self.hparams["latent_variable_intro"] = latent_variable_intro
        self.hparams["lv_prior_mu"] = lv_prior_mu
        self.hparams["lv_prior_std"] = lv_prior_std
        self.hparams["lv_init_std"] = lv_init_std
        self.hparams["lv_latent_dim"] = lv_latent_dim
        self.hparams["init_scaling"] = init_scaling

        self._setup_bnn_with_vi_lv(latent_net)

    def _setup_bnn_with_vi_lv(self, latent_net: nn.Module) -> None:
        """Configure setup of BNN with VI model."""
        # replace the last ultimate layer with nn.Identy so that
        # a user's own model like a `resnet18` that relies on a custom
        # forward pass can still be used as is but we add the final linear
        # layer ourselves
        last_module_name, last_module = _get_output_layer_name_and_module(self.model)
        last_module_args = retrieve_module_init_args(last_module)
        replace_module(self.model, last_module_name, nn.Identity())

        if self.hparams.latent_variable_intro == "first":
            module_name, module = _get_input_layer_name_and_module(self.model)
            new_init_args: dict[str, Union[str, int, float]] = {}
            module_class_name = module.__class__.__name__
            if "Linear" in module_class_name:
                new_init_args["in_features"] = (
                    module.in_features + self.hparams.lv_latent_dim
                )
                lv_init_std = math.sqrt(module.in_features)
            elif "Conv" in module_class_name:
                new_init_args["in_channels"] = (
                    module.in_channels + self.hparams.lv_latent_dim
                )
                lv_init_std = math.sqrt(module.in_channels)
            else:
                raise ValueError

            current_args = retrieve_module_init_args(module)
            current_args.update(new_init_args)
            replace_module(self.model, module_name, module.__class__(**current_args))
        else:
            last_module_args["in_features"] = (
                last_module_args["in_features"] + self.hparams.lv_latent_dim
            )

        # add our own final layer to respect possible custom forward pass
        # import pdb
        # pdb.set_trace()
        self.final_output_module = last_module.__class__(**last_module_args)

        _, lv_output_module = _get_output_layer_name_and_module(latent_net)
        assert lv_output_module.out_features == self.hparams.lv_latent_dim * 2, (
            "The specified latent network needs to have the same output dimension as "
            f"`lv_latent_dim` but found {lv_output_module.out_features} "
            f"and lv_latent_dim {self.hparams.lv_latent_dim}"
        )

        # need to find the output dimension at which latent net is introduced
        self.lv_net = LatentVariableNetwork(
            net=latent_net,
            num_training_points=self.hparams.num_training_points,
            lv_prior_mu=self.hparams.lv_prior_mu,
            lv_prior_std=self.hparams.lv_prior_std,
            lv_init_std=lv_init_std,
            lv_latent_dim=self.hparams.lv_latent_dim,
            init_scaling=self.hparams.init_scaling,
        )

    def forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Forward pass BNN LV.

        Args:
            X: input data
            y: target

        Returns:
            bnn output
        """
        if self.hparams.latent_variable_intro == "first":
            if y is not None:
                # this passes X,y through the whole self.lv_net
                z = self.lv_net(X, y)
            else:
                z = torch.randn(X.shape[0], self.hparams.lv_latent_dim).to(self.device)
            X = torch.cat([X, z], -1)  # [batch_size, n_hidden[?]+1]
            X = self.model(X)
            X = self.final_output_module(X)
        else:
            X = self.model(X)
            # introduce lv
            if y is not None:
                # this passes X,y through the whole self.lv_net
                z = self.lv_net(X, y)
            else:
                z = torch.randn(X.shape[0], self.hparams.lv_latent_dim).to(self.device)
            X = torch.cat([X, z], -1)  # [batch_size, n_hidden[?]+1]
            X = self.final_output_module(X)

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

        # learn output noise 
        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std))**2

        
        # draw samples for all stochastic functions
        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X, y)  # pass X and y during training for lv
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # dim=1
            # collect log f hat from all module parts
            log_f_hat.append(get_log_f_hat([self.model, self.final_output_module]))
            # latent net
            log_f_hat_latent_net.append(self.lv_net.log_f_hat_z)

        # model_preds [num_mc_samples_train, batch_size, output_dim]
        mean_out = torch.stack(model_preds, dim=0).mean(dim=0)

        energy_loss = self.energy_loss_module(
            torch.stack(pred_losses, dim=0),
            torch.stack(log_f_hat, dim=0),
            get_log_Z_prior([self.model, self.final_output_module]),
            get_log_normalizer([self.model, self.final_output_module]),
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
            for i in range(self.hparams.num_mc_samples_test):
                # mean prediction
                pred = self.forward(X)
                pred += torch.randn_like(pred) * torch.exp(self.log_aleatoric_std)
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
