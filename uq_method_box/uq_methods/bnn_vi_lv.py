"""Bayesian Neural Networks with Variational Inference and Latent Variables."""  # noqa: E501

# TODO:
# 2) adjust loss functions such that also a two headed network output trained with nll
# works, and add mse burin-phase as in other modules
# 3) make loss function chooseable to be mse or nll like in other modules
# 7) adapt _build_model function so that
# we define a latent dimension Z neural network
# and a utility function that adds the latent dimension at a desired layer
# e.g. before last activation+linear block

import torch
import torch.nn as nn

from .bnn_vi import BNN_VI


class BNN_LV_VI(BNN_VI):
    """Bayesian Neural Network (BNN) with Latent Variables.

    Trained with Variational Inferece.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
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
        layer_type: str = "reparametrization",
        quantiles: list[float] = [0.1, 0.5, 0.9],
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
                the energy loss
            num_mc_samples_test: number of MC samples during test and prediction
            output_noise_scale: scale of predicted sigmas
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            alpha: alpha divergence parameter
            type: Bayesian layer_type type, "Reparametrization" or "flipout"

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
