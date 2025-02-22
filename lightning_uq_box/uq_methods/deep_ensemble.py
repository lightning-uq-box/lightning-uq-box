# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Implement a Deep Ensemble Model for prediction."""

import os
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor

from .base import BaseModule
from .utils import (
    default_classification_metrics,
    default_px_regression_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    process_classification_prediction,
    process_regression_prediction,
    process_segmentation_prediction,
    save_classification_predictions,
    save_image_predictions,
    save_regression_predictions,
)


class DeepEnsemble(BaseModule):
    """Base Class for different Ensemble Models.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html # noqa: E501
    """

    def __init__(
        self, ensemble_members: list[dict[str, type[LightningModule] | str]]
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            save_dir: path to directory where to store prediction
            quantiles: quantile values to compute for prediction
        """
        super().__init__()
        self.n_ensemble_members = len(ensemble_members)
        self.ensemble_members = ensemble_members

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task."""
        pass

    def forward(self, X: Tensor) -> Tensor:
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
                torch.load(model_config["ckpt_path"], weights_only=True)["state_dict"]
            )
            model_config["base_model"].to(X.device).eval()
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
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key])

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        # save metadata
        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        return out_dict

    def generate_ensemble_predictions(self, X: Tensor) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            the ensemble predictions
        """
        return self.forward(X)  # [batch_size, num_outputs, num_ensemble_members]


class DeepEnsembleRegression(DeepEnsemble):
    """Deep Ensemble Model for regression.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
    """  # noqa: E501

    pred_file_name = "preds.csv"

    def setup_task(self) -> None:
        """Set up task for regression."""
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        pred_dict = process_regression_prediction(preds)
        pred_dict["samples"] = preds
        return pred_dict

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


class DeepEnsembleClassification(DeepEnsemble):
    """Deep Ensemble Model for classification.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
    """  # noqa: E501

    valid_tasks = ["multiclass", "binary", "multilabel"]
    pred_file_name = "preds.csv"

    def __init__(
        self,
        ensemble_members: list[dict[str, type[LightningModule] | str]],
        num_classes: int,
        task: str = "multiclass",
    ) -> None:
        """Initialize a new instance of DeepEnsemble for Classification.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            num_classes: number of classes
            task: classification task, one of "multiclass", "binary" or "multilabel"
        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = num_classes
        super().__init__(ensemble_members)

    def setup_task(self) -> None:
        """Set up task for classification."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        return process_classification_prediction(preds)

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


class DeepEnsembleSegmentation(DeepEnsembleClassification):
    """Deep Ensemble Model for segmentation.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
    """  # noqa: E501

    pred_dir_name = "preds"

    def __init__(
        self,
        ensemble_members: list[dict[str, type[LightningModule] | str]],
        num_classes: int,
        task: str = "multiclass",
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of DeepEnsemble for Segmentation.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            num_classes: number of classes
            task: classification task, one of "multiclass", "binary" or "multilabel"
            save_preds: whether to save predictions
        """
        super().__init__(ensemble_members, num_classes, task)
        self.save_preds = save_preds

    def setup_task(self) -> None:
        """Set up task for segmentation."""
        self.test_metrics = default_segmentation_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        return process_segmentation_prediction(preds)

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir) and self.save_preds:
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)


class DeepEnsemblePxRegression(DeepEnsembleRegression):
    """Deep Ensemble Model for pixelwise regression.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html

    .. versionadded:: 0.2.0
    """  # noqa: E501

    def __init__(
        self,
        ensemble_members: list[dict[str, type[LightningModule] | str]],
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of DeepEnsemble for Pixelwise Regression.

        Args:
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            save_preds: whether to save predictions
        """
        super().__init__(ensemble_members)
        self.save_preds = save_preds

    pred_dir_name = "preds"

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_px_regression_metrics("test")

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir) and self.save_preds:
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader
        """
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)
