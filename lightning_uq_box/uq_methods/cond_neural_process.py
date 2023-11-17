"""Conditional Neural Process."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .base import BaseModule

# TODO: https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/dataset/dataset.py
# TODO: write appropriate Data Generator
# TODO:

# Exlanation Conditional Neural Process
# NPs model the mapping from datasets to parameters directly using Deep Sets Theory
# Encoder: Set Encoder takes in pairs of input and target samples, so all N points in the datasets
#   pass them through an encoder which yields a vector representation over which you take a sum
#   which distills a dataset representation (mapping from dataset to a vector)

# Decoder: function rho, where we also pass in the query location (location at which we want to make a prediction)
#   and together with the encoder representation we pass it through the decoder to get a mean and std
#   representation for each query location (mapping from a vector to mean/var), use decoder to query arbitrary test locations

# Training Procedure


class DeepSensorModule(BaseModule):
    """Lightning Module to train Deep Sensor models."""

    def __init__(self, deep_sensor_model):
        """Initialize a new instance of the Deep Sensor Module.

        Args:
            model: model to train
        """
        super().__init__()

        self.deep_sensor_model = deep_sensor_model
        # this is just the neural process model
        self.np_model = deep_sensor_model.model

        self.fix_noise = None
        self.num_lv_samples = 8
        self.normalise = True

    def forward(self, batch: dict[str, Any]) -> AbstractMultiOutputDistribution:
        """Forward pass of NP Model.

        Args:
            batch: batch containing context, targets sets and model kwargs

        Returns:
            neuralprocess distribution
        """
        return self.np_model(
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

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(self.np_model.parameters(), lr=0.001)
