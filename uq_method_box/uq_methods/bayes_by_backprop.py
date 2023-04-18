"""Bayes By Backprop Model."""

from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from torch import Tensor

from uq_method_box.eval_utils import compute_quantiles_from_std

from .base import BaseModel


class BayesByBackpropModel(BaseModel):
    """Bayes by Backprop Model.

    If you use this model in your research, please cite:

    * https://arxiv.org/abs/1505.05424
    * https://github.com/IntelLabs/bayesian-torch
    """

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        save_dir: str,
        num_mc_samples: int = 30,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bayesian_layer_type: str = "Reparameterization",
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            save_dir: directory path to save
            num_mc_samples:
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `Flipout` or `Reparameterization`
        """
        super().__init__(model_class, model_args, lr, None, save_dir)

        self.bnn_args = {
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu_init": posterior_mu_init,
            "posterior_rho_init": posterior_rho_init,
            "type": bayesian_layer_type,
            "moped_enable": False,
        }
        # convert model to Bayes by Backprop model
        dnn_to_bnn(self.model, self.bnn_args)

        self.num_mc_samples = num_mc_samples
        self.quantiles = quantiles
        self.criterion = nn.MSELoss()

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]
        batch_size = X.shape[0]

        out = self.model(X)
        kl_loss = get_kl_loss(self.model)
        mse_loss = self.criterion(out, y)

        loss = mse_loss + kl_loss / batch_size

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        batch_size = X.shape[0]

        out = self.model(X)
        kl_loss = get_kl_loss(self.model)
        mse_loss = self.criterion(out, y)

        loss = mse_loss + kl_loss / batch_size
        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(self.extract_mean_output(out), y)

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        preds = (
            torch.stack([self.model(X) for _ in range(self.num_mc_samples)], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )  # shape [num_samples, batch_size, num_outputs]

        mean = preds.mean(-1).squeeze(-1)
        std = preds.std(-1).squeeze(-1)
        quantiles = compute_quantiles_from_std(mean, std, self.quantiles)
        return {
            "mean": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
