"""Bayes By Backprop Model."""

from typing import Any, Dict

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
        self, config: Dict[str, Any], model_class: type[nn.Module] = None
    ) -> None:
        """Initialize a new instance of Bayes By Backprop model.

        Args:
            config:
            model: base model to be converted to bayes by backprop model
        """
        super().__init__(config, model_class)

        # get dnn_to_bnn args from dictionary
        self.bayes_bc_backprop_args = self.config["model"]["bayes_by_backprop"]

        # convert model to Bayes by Backprop model
        dnn_to_bnn(self.model, self.bayes_bc_backprop_args)

        self.num_mc_samples = self.config["model"]["mc_samples"]

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

        out = self.forward(X)
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

        out = self.forward(X)
        kl_loss = get_kl_loss(self.model)
        mse_loss = self.criterion(out, y)

        loss = mse_loss + kl_loss / batch_size
        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(self.extract_mean_output(out), y)

        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Test Step."""
        batch = args[0]
        out_dict = self.predict_step(batch[0])
        out_dict["targets"] = batch[1].detach().squeeze(-1).numpy()
        return out_dict

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step via Monte Carlo Sampling.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        preds = (
            torch.stack([self.model(batch) for _ in range(self.num_mc_samples)], dim=-1)
            .detach()
            .numpy()
        )  # shape [num_samples, batch_size, num_outputs]

        mean = preds.mean(-1)
        std = preds.std(-1)
        quantiles = compute_quantiles_from_std(
            mean, std, self.config["model"]["quantiles"]
        )
        return {
            "mean": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
