"""Base Model for UQ methods."""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    default_regression_metrics,
    save_predictions_to_csv,
)


class BaseModule(LightningModule):
    """Define a base module.

    The base module has some basic utilities and attributes
    but is otherwise just an extension of a LightningModule.
    """

    input_key = "input"
    target_key = "target"

    train_metrics = default_regression_metrics("train_")
    val_metrics = default_regression_metrics("val_")
    test_metrics = default_regression_metrics("test_")

    pred_file_name = "predictions.csv"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new instance of the Base Module."""
        super().__init__(*args, **kwargs)

    @property
    def num_inputs(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.model)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_outputs(self.model)


class BaseModel(BaseModule):
    """Deterministic Base Trainer as LightningModule."""

    input_key = "input"
    target_key = "target"

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[torch.optim.Optimizer],
        loss_fn: nn.Module,
        lr_scheduler: type[torch.optim.lr_scheduler.LRScheduler] = None,
        save_dir: str = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: Model Class that can be initialized with arguments from dict,
                or timm backbone name
            lr: learning rate for adam otimizer
            loss_fn: loss function module
            save_dir: directory path to save predictions
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.save_dir = save_dir

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

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        out = self.forward(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])

        self.log("train_loss", loss)  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(self.extract_mean_output(out), batch[self.target_key])

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        out = self.forward(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])

        self.log("val_loss", loss)  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(self.extract_mean_output(out), batch[self.target_key])

        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step."""
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = (
            batch[self.target_key].detach().squeeze(-1).cpu().numpy()
        )

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                out_dict["pred"].squeeze(), batch[self.target_key].squeeze(-1)
            )

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        if "out" in out_dict:
            del out_dict["out"]
        return out_dict

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            out = self.forward(X)
        return {"pred": self.extract_mean_output(out)}

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        if self.save_dir:
            save_predictions_to_csv(
                outputs, os.path.join(self.save_dir, self.pred_file_name)
            )

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
