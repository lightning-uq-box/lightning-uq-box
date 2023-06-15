"""Implement a Deep Ensemble Model for prediction."""

import os
from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor

from .utils import process_model_prediction, save_predictions_to_csv


class DeepEnsembleModel(LightningModule):
    """Base Class for different Ensemble Models."""

    def __init__(
        self,
        ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
        save_dir: str,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            save_dir: path to directory where to store prediction
            quantiles: quantile values to compute for prediction
        """
        super().__init__()
        # make hparams accessible
        self.save_hyperparameters(ignore=["ensemble_members"])
        self.ensemble_members = ensemble_members

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        out: list[torch.Tensor] = []
        for model_config in self.ensemble_members:
            # load the weights into the network
            model_config["base_model"].load_state_dict(
                torch.load(model_config["ckpt_path"])["state_dict"]
            )
            out.append(model_config["base_model"](X))
        return torch.stack(out, dim=-1)

    def test_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Any:
        """Compute test step for deep ensemble and log test metrics.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            dictionary of uncertainty outputs
        """
        X, y = batch
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).numpy()
        return out_dict

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        save_predictions_to_csv(
            outputs, os.path.join(self.hparams.save_dir, "predictions.csv")
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_dim]

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.forward(X).cpu().numpy()

        return process_model_prediction(preds, self.hparams.quantiles)
