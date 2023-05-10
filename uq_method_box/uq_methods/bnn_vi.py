"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

# TODO:
# 1) change dnn_to_bnn function such that only some layers are made stochastic done!
# 2) adjust loss functions such that also a two headed network output trained with nll
# works, and add mse burin-phase as in other modules
# 3) make loss function chooseable to be mse or nll like in other modules


# 5) adapted function based on principles kl_divin bnn_t,
# but with additional dependency on sampled weights yielded by reparameterization trick
#  (additional argument that is stochastic sampled weights)
# to do 5): copy bayesian torch library change layers to include output of term:
# ((var^w - prior_var)/ 2 var^w) * w^2 + (mean^w/var^w)*w
# (just like the kl term but now for computing f(W))
# done for linear layer
# 6) for loss computation include sampling operation
#  in training step (already have this in bnn_vi)
# 7) adapt _build_model function so that
# we define a latent dimension Z neural network
# and a utility function that adds the latent dimension at a desired layer
# e.g. before last activation+linear block
# 8) latent variable network is a BNN with prior variance dependent on input dimension
#  (as a square root of the input dimension to the first stochastic layer chosen)
# concatenate extracted features and output from latent variable BNN
#  and push through stochastic layers to get final output

from typing import Any, Dict, List, Union

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
from .utils import EnergyAlphaDivergence


class BNN_VI(BaseModel):
    """Bayesian Neural Network (BNN) with VI.

    Trained with (VI) Variational Inferece and energy loss.
    """

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
                the energy loss.
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter, set between [1e-6,1].

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

        self.hparams["num_stochastic_modules"] = num_stochastic_modules

    def _setup_bnn_with_vi(self) -> None:
        """Configure setup of the BNN Model."""
        self.bnn_args = {
            "prior_mu": self.hparams.prior_mu,
            "prior_sigma": self.hparams.prior_sigma,
            "posterior_mu_init": self.hparams.posterior_mu_init,
            "posterior_rho_init": self.hparams.posterior_rho_init,
        }
        # convert deterministic model to BNN
        dnn_to_bnnlv_some(
            self.model, self.bnn_args, self.hparams.num_stochastic_modules
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
            torch.tensor([1.0 for _ in range(1)], device=self.device)
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
        output_var = torch.ones_like(y)  # * (torch.exp(self.log_aleatoric_std))

        # the functions sum over the layers of any architecture
        # log_normalizer [1]
        log_normalizer = get_log_normalizer(self.model)
        log_Z_prior = get_log_Z_prior(self.model)

        # draw samples for all stochastic functions
        for i in range(self.hparams.num_mc_samples_train):
            # mean prediction
            pred = self.forward(X)
            model_preds.append(pred)
            # compute prediction loss with nll and track over samples
            # note reduction = "None"
            pred_losses.append(self.nll_loss(pred, y, output_var))
            # dim=1
            log_f_hat.append(get_log_f_hat(self.model))

        # model_preds [num_mc_samples_train, batch_size, output_dim]
        model_preds = torch.stack(model_preds, dim=0)
        # pred_losses [num_mc_samples_train, batch_size, 1]
        pred_losses = torch.stack(pred_losses, dim=0)

        # log_f_hat [num_mc_samples_train, 1]
        log_f_hat = torch.stack(log_f_hat, dim=0)

        mean_out = model_preds.mean(dim=0)

        # TODO once we introduce the latent variable network, compute log_normalizer_z and log_f_hat_z # noqa: E501
        energy_loss = self.energy_loss_module(
            pred_losses,
            log_f_hat,
            log_Z_prior,
            log_normalizer,
            0.0,  # log_normalizer_z
            0.0,  # log_f_hat_z
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
    ) -> Dict[str, np.ndarray]:
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

        mean_out = model_preds.mean(dim=0).squeeze(-1)
        std = model_preds.std(dim=0).squeeze(-1)

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
