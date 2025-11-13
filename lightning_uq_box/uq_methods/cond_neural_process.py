"""Conditional Neural Process."""

from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from neuralprocess.model import Model as NPModel
from neuralprocesses.dist import AbstractMultiOutputDistribution
from torch import Tensor

from .base import BaseModule
from .utils import default_regression_metrics

# TODO: https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/dataset/dataset.py
# TODO: write appropriate Data Generator
# TODO:


class NeuralProcess(BaseModule):
    """Lightning Module to train Deep Sensor models."""

    def __init__(
        self,
        model: NPModel,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ):
        """Initialize a new instance of the Deep Sensor Module.

        Args:
            model: model to train
        """
        super().__init__()

        self.model
        self.fix_noise = None
        self.num_lv_samples = 8
        self.normalise = True

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def forward(self, batch: dict[str, Any]) -> AbstractMultiOutputDistribution:
        """Forward pass of NP Model.

        Args:
            batch: batch containing context, targets sets and model kwargs

        Returns:
            neuralprocess distribution
        """
        return self.model(
            batch["context_data"],
            batch["xt"],
            **batch["model_kwargs"],
            fix_noise=self.fix_noise,
            num_samples=self.num_lv_samples,
            normalise=self.normalise,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Define training step.

        Args:
            batch: a batch from TaskStreamDataset

        Returns:
            the training loss
        """
        train_dist = self.forward(batch)
        train_loss = -torch.mean(train_dist.logpdf(batch["yt"]))

        # logging
        self.log("train_loss", train_loss, batch_size=batch["yt"].shape[0])

        self.train_metrics(
            train_dist.mean.squeeze().reshape(-1), batch["yt"].squeeze().reshape(-1)
        )

        return train_loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Define validation step.

        Args:
            batch: a batch from TaskStreamDataset

        Returns:
            the validation loss
        """
        val_dist = self.forward(batch)
        val_loss = -torch.mean(val_dist.logpdf(batch["yt"]))

        # logging
        self.log("val_loss", val_loss, batch_size=batch["yt"].shape[0])
        self.val_metrics(
            val_dist.mean.squeeze().reshape(-1), batch["yt"].squeeze().reshape(-1)
        )

        return val_loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Define test step.

        Args:
            batch: a batch from TaskStreamDataset

        Returns:
            the test loss
        """
        test_dist = self.forward(batch)
        test_loss = -torch.mean(test_dist.logpdf(batch["yt"]))

        # logging
        self.log("test_loss", test_loss, batch_size=batch["yt"].shape[0])
        self.test_metrics(
            test_dist.mean.squeeze().reshape(-1), batch["yt"].squeeze().reshape(-1)
        )

        return test_loss

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """Prediction Step."""

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
