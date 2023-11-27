# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

import math
from typing import Any, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.models.bnnlv.latent_variable_network import LatentVariableNetwork
from lightning_uq_box.models.bnnlv.utils import (
    get_log_f_hat,
    get_log_normalizer,
    get_log_Z_prior,
    replace_module,
    retrieve_module_init_args,
)
from lightning_uq_box.uq_methods.utils import (
    _get_input_layer_name_and_module,
    _get_output_layer_name_and_module,
    default_regression_metrics,
)

from .bnn_vi import BNN_VI_Base


class BNN_LV_VI_Base(BNN_VI_Base):
    """Bayesian Neural Network (BNN) with Latent Variables (LV).

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a

    """

    lv_intro_options = ["first", "last"]

    def __init__(
        self,
        model: nn.Module,
        latent_net: nn.Module,
        num_training_points: int,
        prediction_head: Optional[nn.Module] = None,
        stochastic_module_names: Optional[list[Union[str, int]]] = None,
        latent_variable_intro: str = "first",
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        n_mc_samples_epistemic: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        bayesian_layer_type: str = "reparameterization",
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 1.0,
        lv_latent_dim: int = 1,
        init_scaling: float = 0.1,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instace of BNN+LV.

        Args:
            model: pytorch model that will be converted into a BNN
            latent_net: latent variable network
            num_training_points: num of data points contained in the training dataset
            num_training_points: num of data points contained in the training dataset
            prediction_head: prediction head that will be attached to the model
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            latent_variable_intro: whether to introduce the latent variable at
                the first or last layer of the model
            n_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss
            n_mc_samples_test: number of MC samples during test and prediction
            n_mc_samples_epistemic: number of epistemic samples during prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            bayesian_layer_type: `flipout` or `reparameterization`
            lv_prior_mu: prior mean for latent variable network
            lv_prior_std: prior std for latent variable network
            lv_latent_dim: number of latent dimension
            init_scaling: init scaling factor for q(z) in latent variable network
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive
            AssertionError: if ``n_mc_samples_test`` is not positive
        """
        super().__init__(
            model,
            num_training_points,
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
            optimizer,
            lr_scheduler,
        )

        assert (
            latent_variable_intro in self.lv_intro_options
        ), f"Only one of {self.lv_intro_options} is possible, but found {latent_variable_intro}."  # noqa: E501

        self.save_hyperparameters(
            ignore=[
                "model",
                "latent_net",
                "prediction_head",
                "optimizer",
                "lr_scheduler",
            ]
        )

        self.prediction_head = prediction_head

        self._setup_bnn_with_vi_lv(latent_net)

    def setup_task(self) -> None:
        """Set up task."""
        pass

    def _setup_bnn_with_vi_lv(self, latent_net: nn.Module) -> None:
        """Configure setup of BNN with VI model.

        Args:
            latent_net: latent variable network
        """
        # replace the last ultimate layer with nn.Identy so that
        # a user's own model like a `resnet18` that relies on a custom
        # forward pass can still be used as is but we add the final linear
        # layer ourselves
        last_module_name, last_module = _get_output_layer_name_and_module(self.model)
        last_module_args = retrieve_module_init_args(last_module)
        replace_module(self.model, last_module_name, nn.Identity())

        if self.hparams.latent_variable_intro == "first":
            module_name, module = _get_input_layer_name_and_module(self.model)

            if "Conv" in module.__class__.__name__:
                raise ValueError(
                    "First layer cannot be Convolutional Layer if "
                    "*latent_variable_intro* is 'first'. Please use 'last' instead."
                )

            lv_init_std = math.sqrt(module.in_features)
            new_init_args: dict[str, Union[str, int, float]] = {}
            new_init_args["in_features"] = (
                module.in_features + self.hparams.lv_latent_dim
            )
            current_args = retrieve_module_init_args(module)
            current_args.update(new_init_args)
            replace_module(self.model, module_name, module.__class__(**current_args))

            # check latent net
            _, lv_input_module = _get_input_layer_name_and_module(latent_net)
            assert (
                lv_input_module.in_features
                == module.in_features + last_module.out_features
            ), (
                "The specified latent network needs to have an input dimension that "
                "is equal to the sum of the dataset features (first layer in_features) "
                "and the target dimension but found latent network input dimension "
                f"of {lv_input_module.in_features} but a sum of "
                f"{module.in_features + last_module.out_features}."
            )
        else:  # last layer
            last_module_args["in_features"] = (
                last_module_args["in_features"] + self.hparams.lv_latent_dim
            )
            lv_init_std = math.sqrt(last_module_args["in_features"])

            module_name, module = _get_input_layer_name_and_module(self.model)
            first_module_args = retrieve_module_init_args(module)
            if "in_features" in first_module_args:
                data_dim = first_module_args["in_features"]  # first layer lin
                test_x = torch.randn(5, 5, data_dim)
            else:
                data_dim = first_module_args["in_channels"]  # first layer conv
                test_x = torch.randn(5, 3, data_dim, data_dim)

            with torch.no_grad():
                feature_output = self.model(test_x)

            _, lv_input_module = _get_input_layer_name_and_module(latent_net)

            assert (
                lv_input_module.in_features
                == last_module_args["out_features"] + feature_output.shape[-1]
            ), (
                "The specified latent network needs to have an input dimension that "
                "is equal to the sum of the feature output dimension of the model and "
                "the target dimension but found latent network input dimension "
                f"of {lv_input_module.in_features} and a feature space output "
                f"of {feature_output.shape[-1]} with a target dimension of "
                f"{last_module_args['out_features']}."
            )

        if not self.prediction_head and self.hparams.latent_variable_intro == "first":
            # keep last module
            self.prediction_head = last_module.__class__(**last_module_args)
        elif not self.prediction_head and self.hparams.latent_variable_intro == "last":
            # provide a default
            self.prediction_head = nn.Sequential(
                nn.Linear(last_module_args["in_features"], 50),
                nn.ReLU(),
                nn.Linear(50, last_module_args["out_features"]),
            )
        else:
            # use existing prediction head
            _, module = _get_input_layer_name_and_module(self.prediction_head)
            assert last_module_args["in_features"] == module.in_features

        _, lv_output_module = _get_output_layer_name_and_module(latent_net)
        assert lv_output_module.out_features == self.hparams.lv_latent_dim * 2, (
            "The specified latent network needs to have the same output dimension as "
            f"`lv_latent_dim` but found {lv_output_module.out_features} "
            f"and 2 * lv_latent_dim {self.hparams.lv_latent_dim}"
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

    def forward(
        self, X: Tensor, y: Optional[Tensor] = None, training: bool = True
    ) -> Tensor:
        """Forward pass BNN LV.

        Args:
            X: input data
            y: target
            training: if yes, smple from lv posterior,
                else use sample from prior or provide z

        Returns:
            bnn output of size [batch_size, output_dim]
        """
        if self.hparams.latent_variable_intro == "first":
            if training:
                # this passes X,y through the whole self.lv_net
                z = self.lv_net(X, y)  # [batch_size, lv_latent_dim]
            else:
                if y is not None:
                    z = y
                else:
                    z = self.sample_latent_variable_prior(X)

            X = torch.cat(
                [X, z], -1
            )  # [batch_size, num_dataset_features+lv_latent_dim]
            X = self.model(X)
            X = self.prediction_head(X)
        else:
            X = self.model(X)
            # introduce lv
            if training:
                # this passes X,y through the whole self.lv_net
                z = self.lv_net(X, y)
            else:
                if y is not None:
                    z = y
                else:
                    z = self.sample_latent_variable_prior(X)

            X = torch.cat(
                [X, z], -1
            )  # [batch_size, model output_features+lv_latent_dim]
            X = self.prediction_head(X)

        return X

    def sample_latent_variable_prior(self, X: Tensor) -> Tensor:
        """Sample the latent variable prior during inference.

        Args:
            X: inference tensor that gets concatenated with z

        Returns:
            sampled latent variable of shape [batch_size, lv_latent_dim]
        """
        batch_size = X.shape[0]
        return torch.randn(batch_size, self.hparams.lv_latent_dim).to(self.device)

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> Tuple[Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
            mean_out: mean output over samples,
                dim [n_mc_samples_train, output_dim]
        """
        model_preds = []
        pred_losses = []
        log_f_hat = []
        log_f_hat_latent_net = []

        # learn output noise
        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2

        # draw samples for all stochastic functions
        for i in range(self.hparams.n_mc_samples_train):
            # mean prediction
            pred = self.forward(X, y)  # pass X and y during training for lv
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # collect log f hat from all module parts
            log_f_hat.append(get_log_f_hat([self.model, self.prediction_head]))
            # latent net
            log_f_hat_latent_net.append(self.lv_net.log_f_hat_z)

        # model_preds [batch_size, output_dim, n_mc_samples_train]
        mean_out = torch.stack(model_preds, dim=-1).mean(dim=-1)

        energy_loss = self.energy_loss_module(
            torch.stack(pred_losses, dim=0),
            torch.concat(log_f_hat, dim=0),
            get_log_Z_prior([self.model, self.prediction_head]),
            get_log_normalizer([self.model, self.prediction_head]),
            self.lv_net.log_normalizer_z,  # log_normalizer_z
            torch.stack(log_f_hat_latent_net, dim=0),  # log_f_hat_z
        )

        return energy_loss, mean_out.detach()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        n_aleatoric = self.hparams.n_mc_samples_epistemic

        if self.hparams.latent_variable_intro == "first":
            output_dim = self.prediction_head.out_features
        else:
            key, module = _get_output_layer_name_and_module(self.prediction_head)
            output_dim = module.out_features

        in_noise = torch.randn(n_aleatoric)
        model_preds_hy = np.zeros(
            (self.hparams.n_mc_samples_epistemic, X.shape[0], output_dim)
        )
        model_preds = np.zeros(
            (self.hparams.n_mc_samples_epistemic, n_aleatoric, X.shape[0], output_dim)
        )
        o_noise = torch.exp(self.log_aleatoric_std).detach().cpu().numpy()
        with torch.no_grad():
            for i in range(self.hparams.n_mc_samples_epistemic):
                self.freeze_layers()
                z = torch.tile(in_noise[i], (X.shape[0], 1))
                pred = self.forward(X, z, training=False).cpu().numpy()
                pred += (
                    np.tile(np.random.randn(1, output_dim), [X.shape[0], 1]) * o_noise
                )
                model_preds_hy[i, :, :] = pred

            for i in range(self.hparams.n_mc_samples_epistemic):
                # one forward pass to resample
                self.freeze_layers()
                for j in range(n_aleatoric):
                    z = torch.tile(in_noise[j], (X.shape[0], 1))
                    pred = self.forward(X, z, training=False).cpu().numpy()
                    pred += (
                        np.tile(np.random.randn(1, output_dim), [X.shape[0], 1])
                        * o_noise
                    )
                    model_preds[i, j, :, :] = pred
                self.unfreeze_layers()

        mean_out = model_preds.mean(axis=(0, 1)).squeeze()

        def entropy(x, axis=None):
            var_x = x.var(axis=axis)
            # clip variance to avoid numerical issues
            var_x = np.clip(var_x, 1e-6, None)
            return 0.5 * np.log(2 * np.pi * var_x) + 0.5

        full_uncertainty = entropy(model_preds_hy, axis=0).ravel()
        aleatoric_uncertainty = entropy(model_preds, axis=1).mean(axis=0).ravel()
        epistemic_uncertainty = full_uncertainty - aleatoric_uncertainty
        std_full = model_preds_hy.std(axis=0).squeeze()

        return {
            "pred": torch.from_numpy(mean_out),
            "pred_uct": std_full,
            "epistemic_uct": epistemic_uncertainty,
            "aleatoric_uct": aleatoric_uncertainty,
            "samples": model_preds_hy,
        }

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

    # TODO optimize both bnn and lv model parameters
    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        # optimizer_args = getattr(self.optimizer, "keywords")
        # wd = optimizer_args.get("weight_decay", 0.0)
        # TODO this does not work with lightning CLI correctly yet
        # self.optimizer is not a partial function anymore that can be accessed
        #  with keywords using default weight decay for now
        params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=0.01)

        optimizer = self.optimizer(params)
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class BNN_LV_VI_Regression(BNN_LV_VI_Base):
    """Bayesian Latent Variable Network with Variational Inference for Regression.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)

    def setup_task(self) -> None:
        """Set up task."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")


class BNN_LV_VI_Batched_Base(BNN_LV_VI_Base):
    """Batched sampling version of BNN_LV_VI.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    def __init__(
        self,
        model: nn.Module,
        latent_net: nn.Module,
        num_training_points: int,
        prediction_head: Optional[nn.Module] = None,
        stochastic_module_names: Optional[list[Union[str, int]]] = None,
        latent_variable_intro: str = "first",
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        n_mc_samples_epistemic: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        bayesian_layer_type: str = "reparameterization",
        lv_prior_mu: float = 0,
        lv_prior_std: float = 1,
        lv_latent_dim: int = 1,
        init_scaling: float = 0.1,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instace of BNN+LV Batched.

        Args:
            model: pytorch model that will be converted into a BNN
            latent_net: latent variable network
            num_training_points: number of data points contained in the training dataset
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            num_training_points: number of data points contained in the training dataset
            prediction_head: prediction head that will be attached to the model
            stochastic_module_names:
            latent_variable_intro:
            n_mc_samples_train: number of MC samples during training when computing
                the negative ELBO loss
            n_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            bayesian_layer_type: `flipout` or `reparameterization`
            lv_prior_mu: prior mean for latent variable network
            lv_prior_std: prior std for latent variable network
            lv_latent_dim: number of latent dimension
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler


        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive
            AssertionError: if ``n_mc_samples_test`` is not positive
        """
        super().__init__(
            model,
            latent_net,
            num_training_points,
            prediction_head,
            stochastic_module_names,
            latent_variable_intro,
            n_mc_samples_train,
            n_mc_samples_test,
            n_mc_samples_epistemic,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            bayesian_layer_type,
            lv_prior_mu,
            lv_prior_std,
            lv_latent_dim,
            init_scaling,
            optimizer,
            lr_scheduler,
        )

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.bayesian_layer_type,
            "batched_samples": True,
            "max_n_samples": max(
                self.hparams.n_mc_samples_train, self.hparams.n_mc_samples_test
            ),
        }

    def forward(
        self,
        X: Tensor,
        y: Optional[Tensor] = None,
        n_samples: int = 5,
        training: bool = True,
    ) -> Tensor:
        """Forward pass BNN+LI.

        Args:
            X: input data
            y: target
            n_samples: number of samples to compute

        Returns:
            bnn output [batch_size, output_dim, num_samples]
        """
        batched_sample_X = einops.repeat(X, "b f -> s b f", s=n_samples)
        if y is not None:
            batched_sample_y = einops.repeat(X, "b f -> s b f", s=n_samples)
        else:
            batched_sample_y = None
        return super().forward(batched_sample_X, batched_sample_y, training=training)

    def sample_latent_variable_prior(self, X: Tensor) -> Tensor:
        """Sample the latent variable prior during inference.

        Args:
            X: inference tensor that gets concatenated with z

        Returns:
            sampled latent variable of shape [batch_size, lv_latent_dim]
        """
        num_samples = X.shape[0]
        batch_size = X.shape[1]
        return torch.randn(num_samples, batch_size, self.hparams.lv_latent_dim).to(
            self.device
        )

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> Tuple[Tensor]:
        """Compute the loss for BNN with alpha divergence.

        Args:
            X: input tensor
            y: target tensor

        Returns:
            energy loss and mean output for logging
            mean_out: mean output over samples,
                dim [n_mc_samples_train, output_dim]
        """
        out = self.forward(
            X, y, n_samples=self.hparams.n_mc_samples_train
        )  # [n_mc_samples_train, batch_size, output_dim]

        y = torch.tile(y[None, ...], (self.hparams.n_mc_samples_train, 1, 1))
        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2

        energy_loss = self.energy_loss_module(
            self.nll_loss(out, y, output_var),
            get_log_f_hat([self.model, self.prediction_head])[
                : self.hparams.n_mc_samples_train
            ],  # noqa: E203
            get_log_Z_prior([self.model, self.prediction_head]),
            get_log_normalizer([self.model, self.prediction_head]),
            log_normalizer_z=self.lv_net.log_normalizer_z,  # log_normalizer_z
            log_f_hat_z=self.lv_net.log_f_hat_z,  # log_f_hat_z
        )
        return energy_loss, out.detach().mean(dim=0)

    def predict_step(
        self,
        X: Tensor,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        n_samples_pred: int = 100,
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        n_aleatoric = self.hparams.n_mc_samples_epistemic
        n_samples = self.hparams.n_mc_samples_test

        if self.hparams.latent_variable_intro == "first":
            output_dim = self.prediction_head.out_features
        else:
            key, module = _get_output_layer_name_and_module(self.prediction_head)
            output_dim = module.out_features

        in_noise = torch.randn(n_aleatoric)
        model_preds_hy = np.zeros(
            (self.hparams.n_mc_samples_epistemic, X.shape[0], output_dim)
        )

        model_preds = np.zeros(
            (self.hparams.n_mc_samples_epistemic, n_aleatoric, X.shape[0], output_dim)
        )
        o_noise = torch.exp(self.log_aleatoric_std).detach().cpu().numpy()

        with torch.no_grad():
            for i in range(int(self.hparams.n_mc_samples_epistemic / n_samples)):
                self.freeze_layers(n_samples)
                z = torch.tile(
                    in_noise[i * n_samples : (i + 1) * n_samples][  # noqa: E203
                        :, None, None
                    ],
                    (1, X.shape[0], 1),
                )
                pred = (
                    super()
                    .forward(
                        torch.tile(X[None, ...], [n_samples, 1, 1]), z, training=False
                    )
                    .cpu()
                    .numpy()
                )
                pred += (
                    np.tile(
                        np.random.randn(n_samples, 1, output_dim), [1, X.shape[0], 1]
                    )
                    * o_noise
                )
                model_preds_hy[
                    i * n_samples : (i + 1) * n_samples, :, :  # noqa: E203
                ] = pred

            for i in range(int(self.hparams.n_mc_samples_epistemic / n_samples)):
                # freeze will resample
                self.freeze_layers(n_samples)
                for j in range(n_aleatoric):
                    z = torch.tile(in_noise[j], (n_samples, X.shape[0], 1))
                    pred = (
                        super()
                        .forward(
                            torch.tile(X[None, ...], [n_samples, 1, 1]),
                            z,
                            training=False,
                        )
                        .cpu()
                        .numpy()
                    )
                    pred += (
                        np.tile(
                            np.random.randn(n_samples, 1, output_dim),
                            [1, X.shape[0], 1],
                        )
                        * o_noise
                    )
                    model_preds[
                        i * n_samples : (i + 1) * n_samples, j, :, :  # noqa: E203
                    ] = pred
                self.unfreeze_layers()

        mean_out = model_preds.mean(axis=(0, 1)).squeeze()

        def entropy(x, axis=None):
            var_x = x.var(axis=axis)
            # clip variance to avoid numerical issues
            var_x = np.clip(var_x, 1e-6, None)
            return 0.5 * np.log(2 * np.pi * var_x) + 0.5

        full_uncertainty = entropy(model_preds_hy, axis=0).ravel()
        aleatoric_uncertainty = entropy(model_preds, axis=1).mean(axis=0).ravel()
        epistemic_uncertainty = full_uncertainty - aleatoric_uncertainty
        std_full = model_preds_hy.std(axis=0).squeeze()

        return {
            "pred": torch.from_numpy(mean_out),
            "pred_uct": std_full,
            "epistemic_uct": epistemic_uncertainty,
            "aleatoric_uct": aleatoric_uncertainty,
            "samples": model_preds_hy,
        }

    def freeze_layers(self, n_samples: int) -> None:
        """Freeze BNN Layers to fix the stochasticity over forward passes.

        Args:
            n_samples: number of samples used in frozen layers
        """
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.freeze_layer(n_samples)


class BNN_LV_VI_Batched_Regression(BNN_LV_VI_Batched_Base):
    """Bayesian Latent Variable Network with VI Batched for Regression.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v80/depeweg18a
    """

    nll_loss = nn.GaussianNLLLoss(reduction="none", full=True)

    def setup_task(self) -> None:
        """Set up task."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")
