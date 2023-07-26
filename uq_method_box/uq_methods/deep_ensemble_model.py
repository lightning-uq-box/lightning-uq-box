"""Implement a Deep Ensemble Model for prediction."""

import os
from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor

from .utils import process_model_prediction, save_predictions_to_csv

from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score


class DeepEnsembleModel(LightningModule):
    """Base Class for different Ensemble Models."""

    def __init__(
        self,
        n_ensemble_members: int,
        ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
        save_dir: str,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            n_ensemble_members: number of ensemble members
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            save_dir: path to directory where to store prediction
            quantiles: quantile values to compute for prediction
        """
        super().__init__()
        assert len(ensemble_members) == n_ensemble_members
        # make hparams accessible
        self.save_hyperparameters(ignore=["ensemble_members"])
        self.ensemble_members = ensemble_members


        self.test_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(),
            },
            prefix="test_",
        )

        self.pred_file_name = "predictions.csv"

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
            model_config["base_model"].to(X.device)
            out.append(model_config["base_model"](X))
        return torch.stack(out, dim=-1)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test step."""
        """Compute test step for deep ensemble and log test metrics.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            dictionary of uncertainty outputs
        """
        out_dict = self.predict_step(batch["inputs"])
        out_dict["targets"] = batch["targets"].detach().squeeze(-1).cpu().numpy()

        # self.log("test_loss", self.loss_fn(out_dict["pred"], batch["targets"].squeeze(-1)))  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch["targets"])

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in ["inputs", "targets"]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        return out_dict

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        if self.hparams.save_dir:
            save_predictions_to_csv(
                outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
            )

    def generate_ensemble_predictions(self, X: Tensor) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            the ensemble predictions
        """
        return self.forward(X)  # [batch_size, num_outputs, num_ensemble_members]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        return process_model_prediction(preds, self.hparams.quantiles)
