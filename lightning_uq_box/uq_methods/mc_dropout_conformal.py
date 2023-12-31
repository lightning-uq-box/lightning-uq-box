# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Mc-Dropout Conformal Prediction."""

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam

from .mc_dropout import MCDropoutClassification, MCDropoutRegression
from .utils import process_classification_prediction, process_regression_prediction


class MCDropoutCPClassification(MCDropoutClassification):
    """MC Dropout with Conformal Prediction for Classification.

    If you use this method, please cite:

    * https://arxiv.org/abs/2308.09647
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        patience: int = 10,
        min_delta: float = 5e-4,
        task: str = "multiclass",
        dropout_layer_names: list[str] = [],
        optimizer: OptimizerCallable = Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize MC Dropout with Conformal Prediction for Classification.

        Args:
            model: PyTorch model
            num_mc_samples: number of MC samples
            loss_fn: loss function
            patience: how many forward passes to wait when all classes are lower than
                the min_delta threshold
            min_delta: considered stable if class value falls below this threshold
            task: task type, either "binary" or "multiclass"
            dropout_layer_names: list of names of dropout layers
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            task,
            dropout_layer_names,
            optimizer,
            lr_scheduler,
        )
        self.patience = patience
        self.min_delta = min_delta

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with MC-Dropout using conformalised procedure.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()
        preds = dynamic_mc_dropout_conformalised(
            self.model, X, self.patience, self.min_delta, self.num_mc_samples
        )
        return process_classification_prediction(preds)


class MCDropoutCPRegression(MCDropoutRegression):
    """MC Dropout with Conformal Prediction for Regression.

    If you use this method, please cite:

    * https://arxiv.org/abs/2308.09647
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        patience: int = 10,
        min_delta: float = 5e-4,
        burnin_epochs: int = 0,
        dropout_layer_names: list[str] = [],
        optimizer: OptimizerCallable = Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize MC Dropout with Conformal Prediction for Regression.

        Args:
            model: PyTorch model
            num_mc_samples: number of MC samples
            loss_fn: loss function
            patience: how many forward passes to wait when all classes are lower than
                the min_delta threshold
            min_delta: considered stable if class value falls below this threshold
            burnin_epochs: number of epochs to wait before switching to loss function
            dropout_layer_names: list of names of dropout layers
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            burnin_epochs,
            dropout_layer_names,
            optimizer,
            lr_scheduler,
        )
        self.patience = patience
        self.min_delta = min_delta

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with MC-Dropout using conformalised procedure.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()
        preds = dynamic_mc_dropout_conformalised(
            self.model, X, self.patience, self.min_delta, self.num_mc_samples
        )
        return process_regression_prediction(preds)


def dynamic_mc_dropout_conformalised(
    model: nn.Module, X: Tensor, patience: int, min_delta: float, max_num_samples: int
) -> Tensor:
    """MC Dropout with Conformal Prediction for Regression.

    If you use this method, please cite:

    * https://arxiv.org/abs/2308.09647

    Args:
        model: PyTorch model
        X: prediction batch of shape [batch_size x input_dims]
        patience: how many forward passes to wait when all classes are lower than
            the min_delta threshold
        min_delta: the threshold a class how to be lower than to be considered stable
        max_num_samples: maximum number of MC samples to collect

    Returns:
        stacked prediction tensor of shape [batch_size x num_outputs x num_mc_samples]
    """
    std_diffs = []
    with torch.no_grad():
        current_patience = 0
        prev_std: Tensor = torch.zeros(1)
        predictions = []
        while current_patience <= patience or len(predictions) == max_num_samples:
            pred = model(X)
            predictions.append(pred)
            std = pred.std(dim=0)
            if len(predictions) != 1:
                std_diff = torch.abs(std - prev_std)
                std_diffs.append(std_diff)
                if torch.all(std_diff <= min_delta):
                    current_patience += 1
                else:
                    current_patience = 0

            prev_std = std

    preds = torch.stack(predictions, dim=-1)
    return preds
