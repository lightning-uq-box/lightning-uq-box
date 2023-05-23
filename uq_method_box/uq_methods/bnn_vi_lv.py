"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

import math
from typing import Any, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchgeo.trainers.utils import _get_input_layer_name_and_module

from uq_method_box.eval_utils import compute_quantiles_from_std
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
        prediction_head: Optional[nn.Module] = None,
        part_stoch_module_names: Optional[list[Union[str, int]]] = None,
        latent_variable_intro: str = "first",
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -5.0,
        alpha: float = 1.0,
        layer_type: str = "reparameterization",
        lv_prior_mu: float = 0.0,
        lv_prior_std: float = 1.0,
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
            prediction_head: prediction head that will be attached to the model
            part_stoch_module_names:
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
            lv_prior_mu: prior mean for latent variable network
            lv_prior_std: prior std for latent variable network
            lv_latent_dim: number of latent dimension
            quantiles: quantiles to compute

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive
            AssertionError: if ``n_mc_samples_test`` is not positive
        """
        super().__init__(
            model,
            optimizer,
            save_dir,
            num_training_points,
            part_stoch_module_names,
            n_mc_samples_train,
            n_mc_samples_test,
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

        self.save_hyperparameters(ignore=["model", "latent_net", "prediction_head"])

        self.prediction_head = prediction_head

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

    def forward(self, X: Tensor, y: Optional[Tensor] = None,training=True) -> Tensor:
        """Forward pass BNN LV.

        Args:
            X: input data
            y: target
            training: if yes, smple from  lv posterior, else use sample from prior or provide z 

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

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> None:
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
            torch.stack(log_f_hat, dim=0),
            get_log_Z_prior([self.model, self.prediction_head]),
            get_log_normalizer([self.model, self.prediction_head]),
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
        # TODO correctly decompose uncertainty epistemic and aleatoric with LV

        n_aleatoric = 100
        n_epistemic = 100
        output_dim = self.prediction_head.out_features
        in_noise = torch.randn(n_aleatoric)

        model_preds = np.zeros((n_epistemic, n_aleatoric, X.shape[0], output_dim))
        with torch.no_grad():
            for i in range(n_epistemic):
                # one forward pass to resample
                _ = self.forward(X,training=False)
                self.freeze_layers()
                for j in range(n_aleatoric):
                    z = torch.tile(in_noise[j], (X.shape[0], 1))
                    model_preds[i, j] = self.forward(X,z,training=False).detach().cpu().numpy()
                self.unfreeze_layers()
        
        mean_out = model_preds.mean(axis=(0, 1)).squeeze()
        std_epistemic = model_preds.mean(axis=1).std(axis=0).squeeze()
        o_noise = torch.exp(self.log_aleatoric_std).detach().cpu().numpy()
        std_aleatoric = np.sqrt(o_noise ** 2 + model_preds.std(axis=1).mean(axis=0) ** 2).squeeze()
        std = np.sqrt(std_epistemic ** 2 + std_aleatoric ** 2)

        # model_preds [n_mc_samples_test, batch_size, output_dim]


        return {"mean": mean_out, "pred_uct": std, "epistemic_uct": std_epistemic, "aleatoric_uct": std_aleatoric}

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
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.weight_decay
        )
        optimizer = self.optimizer(params=params)
        return optimizer


class BNN_LV_VI_Batched(BNN_LV_VI):
    """Batched sampling version of BNN_LV_VI."""

    def __init__(
        self,
        model: nn.Module,
        latent_net: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        save_dir: str,
        num_training_points: int,
        prediction_head: Optional[nn.Module] = None,
        part_stoch_module_names: Optional[list[Union[str, int]]] = None,
        latent_variable_intro: str = "first",
        n_mc_samples_train: int = 25,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        alpha: float = 1,
        layer_type: str = "reparameterization",
        lv_prior_mu: float = 0,
        lv_prior_std: float = 1,
        lv_latent_dim: int = 1,
        init_scaling: float = 0.01,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instace of BNN+LV Batched.

        Args:
            model:
            latent_net: latent variable network
            optimizer:
            save_dir: directory path to save
            num_training_points: number of data points contained in the training dataset
            prediction_head: prediction head that will be attached to the model
            part_stoch_module_names:
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
            lv_prior_mu: prior mean for latent variable network
            lv_prior_std: prior std for latent variable network
            lv_latent_dim: number of latent dimension
            quantiles: quantiles to compute

        Raises:
            AssertionError: if ``n_mc_samples_train`` is not positive
            AssertionError: if ``n_mc_samples_test`` is not positive
        """
        super().__init__(
            model,
            latent_net,
            optimizer,
            save_dir,
            num_training_points,
            prediction_head,
            part_stoch_module_names,
            latent_variable_intro,
            n_mc_samples_train,
            n_mc_samples_test,
            output_noise_scale,
            prior_mu,
            prior_sigma,
            posterior_mu_init,
            posterior_rho_init,
            alpha,
            layer_type,
            lv_prior_mu,
            lv_prior_std,
            lv_latent_dim,
            init_scaling,
            quantiles,
        )

    def _define_bnn_args(self):
        """Define BNN Args."""
        return {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
            "layer_type": self.hparams.layer_type,
            "batched_samples": True,
            "max_n_samples": max(
                self.hparams.n_mc_samples_train, self.hparams.n_mc_samples_test
            ),
        }

    def forward(
        self, X: Tensor, y: Optional[Tensor] = None, n_samples: int = 5, training=True
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
        return super().forward(batched_sample_X, batched_sample_y).permute(1, 2, 0)

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

    def compute_energy_loss(self, X: Tensor, y: Tensor) -> None:
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
            X, n_samples=self.hparams.n_mc_samples_train
        )  # [batch_size, output_dim, n_samples]

        y = einops.repeat(y, "b f -> b f s", s=self.hparams.n_mc_samples_train)

        output_var = torch.ones_like(y) * (torch.exp(self.log_aleatoric_std)) ** 2
        energy_loss = self.energy_loss_module(
            self.nll_loss(out, y, output_var),
            get_log_f_hat([self.model, self.prediction_head]),
            get_log_Z_prior([self.model, self.prediction_head]),
            get_log_normalizer([self.model, self.prediction_head]),
            log_normalizer_z=self.lv_net.log_normalizer_z,  # log_normalizer_z
            log_f_hat_z=self.lv_net.log_f_hat_z,  # log_f_hat_z
        )
        return energy_loss, out.mean(dim=-1)
    '''
    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        model_preds = []

        # output from forward: [n_samples, batch_size, outputs]
        with torch.no_grad():
            model_preds = self.forward(X, self.hparams.n_mc_samples_test)

        mean_out = model_preds.mean(dim=-1).squeeze(-1).cpu().numpy()
        std = model_preds.std(dim=-1).squeeze(-1).cpu().numpy()

        # currently only single output, might want to support NLL output as well
        quantiles = compute_quantiles_from_std(mean_out, std, self.hparams.quantiles)
        return {
            "mean": mean_out,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
    '''
    def freeze_layers(self, n_samples: int) -> None:
        """Freeze BNN Layers to fix the stochasticity over forward passes.

        Args:
            n_samples: number of samples used in frozen layers
        """
        for _, module in self.named_modules():
            if "Variational" in module.__class__.__name__:
                module.freeze_layer(n_samples)
