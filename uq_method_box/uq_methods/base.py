"""Base Model for UQ methods."""

import os
from typing import Any, Dict, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from .utils import retrieve_loss_fn, save_predictions_to_csv


class BaseModel(LightningModule):
    """Deterministic Base Trainer as LightningModule."""

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        loss_fn: str,
        save_dir: str,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model_class: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            model_args: arguments to initialize model_class
            lr: learning rate for adam otimizer
            loss_fn: string name of loss function to use
            save_dir: directory path to save predictions
        """
        super().__init__()
        # makes self.hparams accesible
        self.save_hyperparameters()

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="test_",
        )

        self._build_model()

    def _build_model(self) -> None:
        """Build the underlying model and loss function."""
        # timm model
        if isinstance(self.hparams.model_class, str):
            self.model = timm.create_model(
                self.hparams.model_class, **self.hparams.model_args
            )
        # if own nn module
        else:
            self.model = self.hparams.model_class(**self.hparams.model_args)
            # specific hack for MLP, maybe torchgeo has function that can find
            # first and last layer
            self.n_inputs = self.model.model[0].in_features
            self.n_outputs = self.model.model[-1].out_features

        self.criterion = retrieve_loss_fn(self.hparams.loss_fn)

    def forward(self, X: Tensor, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the model
        """
        return self.model(X, **kwargs)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Different models have different number of outputs, i.e. Gaussian NLL 2
        or quantile regression but for the torchmetrics only
        the mean/median is considered.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert out.shape[-1] <= 2, "Ony support single mean or Gaussian output."
        return out[:, 0:1]

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(self.extract_mean_output(out), y)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).cpu().numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            out = self.forward(X)
        return {
            "mean": self.extract_mean_output(out).squeeze(-1).detach().cpu().numpy()
        }

    def on_test_batch_end(
        self,
        outputs: Dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        save_predictions_to_csv(
            outputs, os.path.join(self.hparams.save_dir, "predictions.csv")
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}
