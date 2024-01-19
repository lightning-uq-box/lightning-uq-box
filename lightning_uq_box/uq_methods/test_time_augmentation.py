# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Time Augmentation (TTA)."""


import os
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .base import PosthocBase
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    process_classification_prediction,
    process_regression_prediction,
    process_segmentation_prediction,
    save_classification_predictions,
    save_regression_predictions,
)

# TODO default augmentations should be composed with Kornia
# and they should also be suggested to be used


class TTABase(PosthocBase):
    """Test Time Augmentation Module.

    In addition to a prediction with no test time augmentation,
    an additional prediction will be made for each element in `tt_augmentation`.
    """

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        tt_augmentation: Optional[list[Callable]] = None,
    ) -> None:
        """Initialize a new instance of TTA module.

        Args:
            model: LightningModule or nn.Module used for prediction
            tt_augmentation: list of test time augmentation function, assumed to accept an
                input that is a dictionary with key `self.input_key` and `self.target_key`
                which is `input` and `target`, if None, a set of default augmentations will
                be used
        """
        super().__init__(model)

        self.tt_augmentation = tt_augmentation

        self.setup_task()

    def setup_task(self) -> None:
        """Setup task."""
        raise NotImplementedError

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No validation step in TTA."""
        pass

    def on_validation_start(self) -> None:
        """No validation step in TTA."""
        pass

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step for TTA procedure."""

    def predict_step(
        self,
        X: Tensor,
        aug: list[Callable] = None,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Predict step with TTA applied.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            aug: augmentation function to apply to X
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            logits and conformalized prediction sets
        """
        self.eval()

        def yield_prediction(X: Tensor) -> Union[Tensor, dict[str, Tensor]]:
            """Yield prediction depending on underlying model."""
            with torch.no_grad():
                if hasattr(self.model, "predict_step"):
                    pred = self.model.predict_step(X)["pred"]
                else:
                    pred = self.model(X)
            return pred

        aug_predictions: Union[list[Tensor], list[dict[str, Tensor]]] = []
        if aug is None:
            aug = self.tt_augmentation
        # first prediction with no augmentation
        aug_predictions.append(yield_prediction(X))

        # iterate over augmentation functions
        for aug_fn in aug:
            X = aug_fn(X)
            aug_predictions.append(yield_prediction(X))

        # combine predictions to common tensor
        if isinstance(aug_predictions[0], dict):
            aug_preds = {}
            for key in aug_predictions[0].keys():
                aug_preds[key] = torch.stack([pred[key] for pred in aug_predictions])
        else:
            aug_preds = {"pred": torch.stack(aug_predictions)}

        return aug_preds


class TTARegression(TTABase):
    """Regression Test Time Augmentation Module."""

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        tt_augmentation: Optional[list[Callable[..., Any]]] = None,
    ) -> None:
        """Initialize a new instance of TTA Regression module.

        Args:
            model: LightningModule or nn.Module used for prediction
            tt_augmentation: list of test time augmentation function, assumed to accept an
                input that is a dictionary with key `self.input_key` and `self.target_key`
                which is `input` and `target`, if None, a set of default augmentations will
                be used
        """
        super().__init__(model, tt_augmentation)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_regression_metrics("test")

        if self.tt_augmentation is None:
            self.tt_augmentation = []

    def predict_step(
        self,
        X: Tensor,
        aug: list[Callable[..., Any]] = None,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Predict step with TTA applied.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            aug: augmentation function to apply to X
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            regression predictions
        """
        aug_preds = super().predict_step(X, aug, batch_idx, dataloader_idx)

        return process_regression_prediction(aug_preds)

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class TTAClassification(TTABase):
    """Classification Test Time Augmentation Module."""

    pred_file_name = "preds.csv"

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        tt_augmentation: Optional[list[Callable[..., Any]]] = None,
        task: str = "multiclass",
    ) -> None:
        """Initialize a new instance of TTA Classification module.

        Args:
            model: LightningModule or nn.Module used for prediction
            tt_augmentation: list of test time augmentation function, assumed to accept an
                input that is a dictionary with key `self.input_key` and `self.target_key`
                which is `input` and `target`, if None, a set of default augmentations will
                be used
            task: task type, one of "binary", "multiclass", "multilabel"
        """
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        self.num_classes = _get_num_outputs(model)
        super().__init__(model, tt_augmentation)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

        if self.tt_augmentation is None:
            self.tt_augmentation = []

    def predict_step(
        self,
        X: Tensor,
        aug: list[Callable[..., Any]] = None,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Predict step with TTA applied.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            aug: augmentation function to apply to X
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            classification predictions
        """
        aug_preds = super().predict_step(X, aug, batch_idx, dataloader_idx)
        return process_classification_prediction(aug_preds)

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )
