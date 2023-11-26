# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Implement a Deep Ensemble Model for prediction."""

from typing import Any, Union

import torch
from lightning import LightningModule
from torch import Tensor

from .base import BaseModule
from .utils import (
    default_classification_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    process_classification_prediction,
    process_regression_prediction,
    process_segmentation_prediction,
)


class DeepEnsemble(BaseModule):
    """Base Class for different Ensemble Models.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html # noqa: E501
    """

    def __init__(
        self,
        n_ensemble_members: int,
        ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
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

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task."""
        pass

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
        out_dict = self.predict_step(batch[self.input_key])
        out_dict["targets"] = batch[self.target_key].detach().squeeze(-1).cpu().numpy()

        # self.log("test_loss", self.loss_fn(out_dict["pred"],
        # batch[self.target_key].squeeze(-1)))
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key])

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

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

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html # noqa: E501
    """

    def setup_task(self) -> None:
        """Set up task for regression."""
        self.test_metrics = default_regression_metrics("test")

    # def on_test_batch_end(
    #     self,
    #     outputs: dict[str, np.ndarray],
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx=0,
    # ):
    #     """Test batch end save predictions."""
    #     if self.hparams.save_dir:
    #         save_predictions_to_csv(
    #             outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
    #         )

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

        return process_regression_prediction(preds)


class DeepEnsembleClassification(DeepEnsemble):
    """Deep Ensemble Model for classification.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html # noqa: E501
    """

    valid_tasks = ["multiclass", "binary", "multilabel"]

    def __init__(
        self,
        n_ensemble_members: int,
        ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
        num_classes: int,
        task: str = "multiclass",
    ) -> None:
        """Initialize a new instance of DeepEnsemble for Classification.

        Args:
            n_ensemble_members: number of ensemble members
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint
            num_classes: number of classes
            task: classification task, one of "multiclass", "binary" or "multilabel"
        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = num_classes
        super().__init__(n_ensemble_members, ensemble_members)

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
            X: input tensor of shape [batch_size, input_di]

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        return process_classification_prediction(preds)


class DeepEnsembleSegmentation(DeepEnsembleClassification):
    """Deep Ensemble Model for segmentation.

    If you use this model in your work, please cite:

    * https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html # noqa: E501
    """

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

        Returns:
            mean and standard deviation of MC predictions
        """
        with torch.no_grad():
            preds = self.generate_ensemble_predictions(X)

        return process_segmentation_prediction(preds)
