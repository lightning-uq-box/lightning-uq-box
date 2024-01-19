# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Mc-Dropout Conformal Prediction."""

import torch
import torch.nn as nn
from torch import Tensor

from .conformal_qr import ConformalQR
from .mc_dropout import activate_dropout_recursive
from .raps import RAPS
from .temp_scaling import temp_scale_logits


class MCDropoutCPClassification(RAPS):
    """MC Dropout with Conformal Prediction for Classification.

    If you use this method, please cite:

    * https://arxiv.org/abs/2308.09647
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        patience: int = 10,
        min_delta: float = 5e-4,
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: int = 5,
        lamda_param: float = 0.01,
        randomized: bool = False,
        allow_zero_sets: bool = False,
        pct_param_tune: float = 0.3,
        lamda_criterion: str = "size",
        task: str = "multiclass",
        dropout_layer_names: list[str] = [],
    ) -> None:
        """Initialize MC Dropout with Conformal Prediction for Classification.

        Args:
            model: PyTorch model
            num_mc_samples: number of MC samples
            patience: how many forward passes to wait when all classes are lower than
                the min_delta threshold
            min_delta: considered stable if class value falls below this threshold
            optim_lr: learning rate for optimizer
            max_iter: maximum number of iterations to run optimizer
            alpha: 1 - alpha is the desired coverage
            kreg: regularization param (smaller kreg leads to smaller sets)
            lamda_param: regularization param (larger lamda leads to smaller sets)
                (any value of kreg and lambda will lead to coverage, but will yield
                different set sizes)
            randomized: whether to use randomized version of conformal prediction
            allow_zero_sets: whether to allow sets of size zero
            pct_param_tune: fraction of calibration data to use for parameter tuning
            lamda_criterion: optimize for 'size' or 'adaptiveness'
            task: task type, one of 'binary', 'multiclass', or 'multilabel'
            dropout_layer_names: list of names of dropout layers
        """
        super().__init__(
            model,
            optim_lr,
            max_iter,
            alpha,
            kreg,
            lamda_param,
            randomized,
            allow_zero_sets,
            pct_param_tune,
            lamda_criterion,
            task,
        )
        self.min_delta = min_delta
        self.patience = patience
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.num_mc_samples = num_mc_samples
        self.dropout_layer_names = dropout_layer_names

    def activate_dropout(self) -> None:
        """Activate dropout layers."""
        activate_dropout_recursive(self.model, self.dropout_layer_names)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with MC-Dropout using conformalised procedure.

        Algorithm 2 in the paper.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()
        preds = dynamic_mc_dropout_conformalised(
            self.model, X, self.patience, self.min_delta, self.num_mc_samples
        ).mean(-1)
        scores = temp_scale_logits(preds, self.temperature)
        S = self.adjust_model_logits(preds)

        return {"pred": scores, "pred_set": S, "logits": scores}


class MCDropoutCPRegression(ConformalQR):
    """MC Dropout with Conformal Prediction for Regression.

    If you use this method, please cite:

    * https://arxiv.org/abs/2308.09647
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        patience: int = 10,
        min_delta: float = 5e-4,
        dropout_layer_names: list[str] = [],
        quantiles: list[float] = [0.1, 0.5, 0.9],
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
            quantiles: quantiles to be used for CQR
        """
        super().__init__(model, quantiles)
        self.num_mc_samples = num_mc_samples
        self.dropout_layer_names = dropout_layer_names
        self.patience = patience
        self.min_delta = min_delta

    def activate_dropout(self) -> None:
        """Activate dropout layers."""
        activate_dropout_recursive(self.model, self.dropout_layer_names)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with MC-Dropout using conformalised procedure.

        Algorithm 3 in the paper.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()
        preds = dynamic_mc_dropout_conformalised(
            self.model, X, self.patience, self.min_delta, self.num_mc_samples
        ).mean(-1)

        mc_cqr_sets = self.adjust_model_logits(preds)

        return {
            "pred": mc_cqr_sets[:, 1],
            "lower_quant": mc_cqr_sets[:, 0],
            "upper_quant": mc_cqr_sets[:, -1],
            "out": mc_cqr_sets,
        }


def dynamic_mc_dropout_conformalised(
    model: nn.Module, X: Tensor, patience: int, min_delta: float, max_num_samples: int
) -> Tensor:
    """MC Dropout with Conformal Prediction for Regression.

    Algorithm 1 in the paper https://arxiv.org/abs/2308.09647.

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
