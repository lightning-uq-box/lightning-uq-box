# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Base Model for UQ methods."""

import os
from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    default_classification_metrics,
    default_px_regression_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    freeze_model_backbone,
    freeze_segmentation_model,
    process_classification_prediction,
    process_segmentation_prediction,
    save_classification_predictions,
    save_image_predictions,
    save_regression_predictions,
)


class BaseModule(LightningModule):
    """Define a base module.

    The base module has some basic utilities and attributes
    but is otherwise just an extension of a LightningModule.

    This is for things useful across all tasks and methods
    """

    input_key = "input"
    target_key = "target"

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

    def add_aux_data_to_dict(
        self, out_dict: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Add auxiliary data to output dictionary.

        Args:
            out_dict: output dictionary
            batch: batch of data

        Returns:
            updated dict
        """
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                if isinstance(val, Tensor):
                    out_dict[key] = val.detach().squeeze(-1)
                else:
                    out_dict[key] = val
        return out_dict


class DeterministicModel(BaseModule):
    """Deterministic Base Trainer as LightningModule."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            loss_fn: loss function used for optimization
            freeze_backbone: whether to freeze the model backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.freeze_backbone = freeze_backbone

        self.setup_task()

        self.freeze_model()

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            freeze_model_backbone(self.model)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        raise NotImplementedError

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out[:, 0:1]

    def forward(self, X: Tensor) -> Any:
        """Forward pass of the model.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the model
        """
        return self.model(X)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        out = self.forward(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )

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
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        out = self.forward(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])

        self.log(
            "val_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dictionary
        """
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(out_dict["pred"]), batch[self.target_key]
            )

        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        if "out" in out_dict:
            del out_dict["out"]

        return out_dict

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            out = self.forward(X)
        return {"pred": self.adapt_output_for_metrics(out)}

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class DeterministicRegression(DeterministicModel):
    """Deterministic Base Trainer for regression as LightningModule."""

    pred_file_name = "preds.csv"

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

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


class DeterministicClassification(DeterministicModel):
    """Deterministic Base Trainer for classification as LightningModule."""

    pred_file_name = "preds.csv"

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        task: str = "multiclass",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deterministic Classification Model.

        Args:
            model: pytorch model
            loss_fn: loss function used for optimization
            task: what kind of classification task, choose one of
                ["binary", "multiclass", "multilabel"]
            freeze_backbone: whether to freeze the model backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.num_classes = _get_num_outputs(model)
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            out = self.forward(X)

        def identity(x, dim=None):
            return x

        return process_classification_prediction(out, aggregate_fn=identity)

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


class DeterministicSegmentation(DeterministicClassification):
    """Deterministic Base Trainer for segmentation as LightningModule."""

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        task: str = "multiclass",
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new Deterministic Segmentation Model.

        Args:
            model: pytorch model
            loss_fn: loss function used for optimization
            task: what kind of classification task, choose one of
                ["binary", "multiclass", "multilabel"]
            freeze_backbone: whether to freeze the model backbone, by default this is
                supported for torchseg Unet models
            freeze_decoder: whether to freeze the model decoder, by default this is
                supported for torchseg Unet models
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder
        super().__init__(model, loss_fn, task, freeze_backbone, optimizer, lr_scheduler)

        self.save_preds = save_preds

    def setup_task(self) -> None:
        """Set up task specific attributes for segmentation."""
        self.train_metrics = default_segmentation_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_segmentation_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_segmentation_metrics(
            "test", self.task, self.num_classes
        )

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        freeze_segmentation_model(self.model, self.freeze_backbone, self.freeze_decoder)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            out = self.forward(X)

        def identity(x, dim=None):
            return x

        return process_segmentation_prediction(out, aggregate_fn=identity)

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


class DeterministicPixelRegression(DeterministicRegression):
    """Deterministic Base Trainer for pixel regression as LightningModule."""

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of Deterministic Pixel Regression.

        Args:
            model: pytorch model
            loss_fn: loss function used for optimization
            freeze_backbone: whether to freeze the model backbone
            freeze_decoder: whether to freeze the model decoder
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        self.freeze_decoder = freeze_decoder
        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)
        self.save_preds = save_preds

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        freeze_segmentation_model(self.model, self.freeze_backbone, self.freeze_decoder)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_px_regression_metrics("train")
        self.val_metrics = default_px_regression_metrics("val")
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
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)


class PosthocBase(BaseModule):
    """Posthoc Base Model for UQ methods."""

    def __init__(self, model: LightningModule | nn.Module) -> None:
        """Initialize a new Post hoc Base Model."""
        super().__init__()

        self.model = model

        self.post_hoc_fitted = False

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        if isinstance(self.model, LightningModule):
            return _get_num_inputs(self.model.model)
        else:
            return _get_num_inputs(self.model)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension to model
        """
        if isinstance(self.model, LightningModule):
            return _get_num_outputs(self.model.model)
        else:
            return _get_num_outputs(self.model)

    def on_train_start(self) -> None:
        """Set model to eval mode."""
        self.eval()
        self.model_logits = []
        self.labels = []

    @torch.no_grad()
    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Single gathering step of model logits and targets.

        Args:
            batch: batch of data
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            underlying model output and labels
        """
        if self.trainer.current_epoch > 0:
            raise RuntimeError(
                "Post-Hoc methods only need one pass over the calibration data loader."
            )
        # optimization procedures later on but need fixed model here
        self.eval()
        self.model_logits.append(self.model(batch[self.input_key]))
        self.labels.append(batch[self.target_key])

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology."""
        raise NotImplementedError

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def adjust_model_logits(self, model_output: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of Posthoc model that adjusts model logits.

        Args:
            X: input tensor of shape [batch_size x input_dims]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, "
                "please call "
                "trainer.fit(model, train_dataloaders=dm.calib_dataloader()) first."
            )

        # predict with underlying model
        with torch.no_grad():
            model_preds: dict[str, Tensor] = self.model(X)

        return self.adjust_model_logits(model_preds)

    def configure_optimizers(self) -> Any:
        """Configure optimizers for posthoc fitting."""
        pass
