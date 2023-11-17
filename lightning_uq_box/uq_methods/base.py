"""Base Model for UQ methods."""

import os
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    save_predictions_to_csv,
)


class BaseModule(LightningModule):
    """Define a base module.

    The base module has some basic utilities and attributes
    but is otherwise just an extension of a LightningModule.

    This is for things useful across all tasks and methods
    """

    input_key = "input"
    target_key = "target"

    pred_file_name = "predictions.csv"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new instance of the Base Module."""
        super().__init__(*args, **kwargs)

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.model)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension to model
        """
        return _get_num_outputs(self.model)


class DeterministicModel(BaseModule):
    """Deterministic Base Trainer as LightningModule."""

    input_key = "input"
    target_key = "target"

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        loss_fn: nn.Module,
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            loss_fn: loss function used for optimization
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

        self.setup_task()

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        raise NotImplementedError

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model output.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out[:, 0:1]

    def forward(self, X: Tensor, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the model
        """
        return self.model(X, **kwargs)

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
        # UNIQUE to method
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

    # def on_test_batch_end(
    #     self,
    #     outputs: dict[str, np.ndarray],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx=0,
    # ):
    #     """Test batch end save predictions."""
    #     if self.save_dir:
    #         save_predictions_to_csv(
    #             outputs, os.path.join(self.save_dir, self.pred_file_name)
    # )

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


class DeterministicRegression(DeterministicModel):
    """Deterministic Base Trainer for regression as LightningModule."""

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")


class DeterministicClassification(DeterministicModel):
    """Deterministic Base Trainer for classification as LightningModule."""

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        loss_fn: nn.Module,
        task: str = "multiclass",
        lr_scheduler: type[LRScheduler] = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            loss_fn: loss function used for optimization
            task: what kind of classification task, choose one of ["binary", "multiclass", "multilabel"]
            lr_scheduler: learning rate scheduler
        """
        self.num_classes = _get_num_outputs(model)
        assert task in self.valid_tasks
        self.task = task
        super().__init__(model, optimizer, loss_fn, lr_scheduler)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract mean output from model output.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )


class PosthocBase(BaseModule):
    def __init__(self, model: Union[LightningModule, nn.Module]) -> None:
        """Initialize a new Post hoc Base Model."""
        super().__init__()

        self.model = model

        self.post_hoc_fitted = False

    def training_step(self, *args: Any, **kwargs: Any):
        """Posthoc Methods do not have a training step."""
        pass

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Postoc Method can use this method to iterate over dataloader."""
        raise NotImplementedError

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step after running posthoc fitting methodology."""
        raise NotImplementedError

    def adjust_model_output(self, model_output: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of CQR model.

        Args:
            X: input tensor of shape [batch_size x input_dims]
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, please call trainer.validate(model, datamodule) first."
            )

        # predict with underlying model
        with torch.no_grad():
            model_preds: dict[str, np.ndarray] = self.model(X)

        return self.adjust_model_output(model_preds)
