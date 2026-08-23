# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Mc-Dropout module."""

import os
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from ._deprecated import warn_legacy_adapter
from .base import Deterministic, DeterministicModel
from .method_specs import MCDROPOUT_SPEC
from .tasks import ClassificationTask, TaskSpec
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_px_regression_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    freeze_model_backbone,
    freeze_segmentation_model,
    process_classification_prediction,
    process_regression_prediction,
    process_segmentation_prediction,
    save_classification_predictions,
    save_image_predictions,
    save_regression_predictions,
)


def find_dropout_layers(model: nn.Module) -> list[str]:
    """Find dropout layers in model."""
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append(name)

    return dropout_layers


class MCDropoutBase(DeterministicModel):
    """MC-Dropout Base class.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        dropout_layer_names: list[str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instance of MCDropoutModel.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            dropout_layer_names: names of dropout layers to activate during prediction
            freeze_backbone: freeze backbone during training
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        if dropout_layer_names is None:
            dropout_layer_names = []
        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

        if not dropout_layer_names:
            dropout_layer_names = find_dropout_layers(model)
        self.dropout_layer_names = dropout_layer_names

    def setup_task(self) -> None:
        """Set up task specific attributes."""

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
        self.train_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def activate_dropout(self) -> None:
        """Activate dropout layers."""
        dropout_layers_found = []
        self.model.train()

        def activate_dropout_recursive(model, prefix=""):
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in self.dropout_layer_names and isinstance(
                    module, nn.Dropout
                ):
                    module.train()
                    dropout_layers_found.append(full_name)
                elif isinstance(module, nn.Module):
                    activate_dropout_recursive(module, full_name)
                # set batch norm layers to eval mode
                elif isinstance(
                    module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
                ):
                    module.eval()

        activate_dropout_recursive(self.model)

        if not dropout_layers_found:
            raise UserWarning(
                "No dropout layers found in model, maybe dropout "
                "is implemented via specialized layers?"
            )


class MCDropoutRegression(MCDropoutBase):
    """Deprecated MC-Dropout regression compatibility adapter.

    .. versionchanged:: 0.4.0

       Use :class:`MCDropout` with :class:`RegressionTask` for new code. This
       adapter retains its historical constructor, sampling behavior, and
       state-dict prefixes through 0.4.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        burnin_epochs: int = 0,
        dropout_layer_names: list[str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Regression.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            burnin_epochs: number of burnin epochs before using the loss_fn
            dropout_layer_names: names of dropout layers to activate during prediction
            freeze_backbone: freeze backbone during training
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
                from the predictive distribution
        """
        if type(self) is MCDropoutRegression:
            warn_legacy_adapter(
                "MCDropoutRegression", "MCDropout(..., task=RegressionTask())"
            )
        if dropout_layer_names is None:
            dropout_layer_names = []
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            dropout_layer_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            freeze_model_backbone(self.model)

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.."""
        assert out.shape[-1] <= 2, "Ony support single mean or Gaussian output."
        return out[:, 0:1]

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

        if self.current_epoch < self.hparams.burnin_epochs:
            loss = nn.functional.mse_loss(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )
        else:
            loss = self.loss_fn(out, batch[self.target_key])

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        self.train_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_regression_prediction(preds)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class MCDropoutClassification(MCDropoutBase):
    """Deprecated MC-Dropout classification compatibility adapter.

    .. versionchanged:: 0.4.0

       Use :class:`MCDropout` with :class:`ClassificationTask` for new code.
       This adapter preserves its string task argument and state-dict prefixes
       through 0.4.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.mlr.press/v48/gal16.html
    """

    pred_file_name = "preds.csv"
    valid_tasks: ClassVar[list[str]] = ["binary", "multiclass", "multilabel"]

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        task: str = "multiclass",
        dropout_layer_names: list[str] | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Classification.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            dropout_layer_names: names of dropout layers to activate during prediction
            freeze_backbone: freeze backbone during training
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        if type(self) is MCDropoutClassification:
            warn_legacy_adapter(
                "MCDropoutClassification",
                "MCDropout(..., task=ClassificationTask(...))",
            )
        if dropout_layer_names is None:
            dropout_layer_names = []
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = _get_num_outputs(model)
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            dropout_layer_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
        )
        # FIXME: why isn't save_hyperparameters working?
        self.num_mc_samples = num_mc_samples

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

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Extract mean output from model."""
        return out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()  # activate dropout during prediction
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_classification_prediction(preds, task=self.task)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs,
            os.path.join(self.trainer.default_root_dir, self.pred_file_name),
            task=self.task,
        )


class MCDropoutSegmentation(MCDropoutClassification):
    """Deprecated MC-Dropout segmentation compatibility adapter.

    .. versionchanged:: 0.4.0

       Use :class:`MCDropout` with :class:`SegmentationTask` for new code.
       This adapter retains dense prediction persistence and its historical
       constructor/state-dict surface through 0.4.
    """

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        task: str = "multiclass",
        dropout_layer_names: list[str] | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Segmentation.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            dropout_layer_names: names of dropout layers to activate during prediction
            freeze_backbone: whether to freeze the model backbone, by default this is
                supported for SMP Unet models
            freeze_decoder: whether to freeze the model decoder, by default this is
                supported for SMP Unet models
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        if type(self) is MCDropoutSegmentation:
            warn_legacy_adapter(
                "MCDropoutSegmentation", "MCDropout(..., task=SegmentationTask(...))"
            )
        if dropout_layer_names is None:
            dropout_layer_names = []
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            task,
            dropout_layer_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

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
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x num_channels x height x width]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            mean and standard deviation of MC predictions
        """
        self.activate_dropout()  # activate dropout during prediction
        with torch.no_grad():
            preds = torch.stack(
                [self.model(X) for _ in range(self.hparams.num_mc_samples)], dim=-1
            )  # shape [batch_size, num_outputs, num_samples]

        return process_segmentation_prediction(preds, task=self.task)

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir) and self.save_preds:
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
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


class MCDropoutPxRegression(MCDropoutRegression):
    """MC-Dropout Model for Pixel-wise Regression.

    .. versionadded:: 0.2.0

    .. versionchanged:: 0.4.0

       Use :class:`MCDropout` with :class:`PixelRegressionTask` for new code.
       This adapter preserves dense persistence and the historical state-dict
       topology through 0.4.
    """

    pred_dir_name = "preds"

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        burnin_epochs: int = 0,
        dropout_layer_names: list[str] | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a new instance of MC-Dropout Model for Pixel-wise Regression.

        Args:
            model: pytorch model with dropout layers
            num_mc_samples: number of MC samples during prediction
            loss_fn: loss function
            burnin_epochs: number of burnin epochs before using the loss_fn
            dropout_layer_names: names of dropout layers to activate during prediction
            freeze_backbone: freeze backbone during training
            freeze_decoder: freeze decoder during training
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            save_preds: whether to save predictions
        """
        if type(self) is MCDropoutPxRegression:
            warn_legacy_adapter(
                "MCDropoutPxRegression", "MCDropout(..., task=PixelRegressionTask())"
            )
        if dropout_layer_names is None:
            dropout_layer_names = []
        self.freeze_decoder = freeze_decoder
        super().__init__(
            model,
            num_mc_samples,
            loss_fn,
            burnin_epochs,
            dropout_layer_names,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )
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

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.."""
        assert out.shape[1] <= 2, "Ony support single mean or Gaussian output."
        return out[:, 0:1, ...].contiguous()

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir) and self.save_preds:
            os.makedirs(self.pred_dir)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
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


class MCDropout(Deterministic):
    """Canonical MC Dropout method parameterized by an explicit task value.

    The historical ``MCDropoutRegression``, ``MCDropoutClassification``,
    ``MCDropoutSegmentation``, and ``MCDropoutPxRegression`` classes remain
    unchanged as 0.4 compatibility entry points.  This class owns stochastic
    sampling and aggregation while the canonical runtime owns only task
    handling, metrics, result shaping, and persistence.

    .. versionchanged:: 0.4.0

       MC Dropout now has one canonical method API with an explicit task
       value. Standard dropout activation and stochastic aggregation remain
       method-owned; tasks never infer a distribution from output shape.
    """

    method_spec = MCDROPOUT_SPEC

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        loss_fn: nn.Module,
        *,
        task: TaskSpec | Mapping[str, Any] | None = None,
        dropout_layer_names: list[str] | None = None,
        burnin_epochs: int = 0,
        prediction_kind: str = "point",
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize canonical MC Dropout.

        Args:
            model: model containing named ``nn.Dropout`` modules.
            num_mc_samples: number of stochastic forward passes for prediction.
            loss_fn: method-owned training loss.
            task: immutable task value or supported task configuration mapping.
            dropout_layer_names: dropout module names to activate.  ``None``
                discovers all standard ``nn.Dropout`` modules.
            burnin_epochs: initial point-loss epochs for regression workflows.
            prediction_kind: ``"point"`` or explicit ``"gaussian"`` output
                conversion.  This is method-owned and never guessed by a task.
            freeze_backbone: freeze the model backbone before training.
            freeze_decoder: freeze a dense-model decoder before training.
            optimizer: optimizer factory.
            lr_scheduler: optional learning-rate scheduler factory.
            save_preds: persist dense predictions in test hooks.
        """
        if num_mc_samples < 1:
            raise ValueError("num_mc_samples must be at least one.")
        if prediction_kind not in {"point", "gaussian"}:
            raise ValueError("prediction_kind must be 'point' or 'gaussian'.")
        self.num_mc_samples = num_mc_samples
        self.dropout_layer_names = (
            list(dropout_layer_names)
            if dropout_layer_names is not None
            else find_dropout_layers(model)
        )
        self.burnin_epochs = burnin_epochs
        self.prediction_kind = prediction_kind
        super().__init__(
            model,
            loss_fn,
            task=task,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            save_preds=save_preds,
        )
        self.save_hyperparameters(
            {
                "num_mc_samples": num_mc_samples,
                "dropout_layer_names": self.dropout_layer_names,
                "burnin_epochs": burnin_epochs,
                "prediction_kind": prediction_kind,
            }
        )

    def activate_dropout(self) -> None:
        """Enable exactly the configured dropout modules for MC prediction."""
        found: set[str] = set()
        self.model.train()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
            if name in self.dropout_layer_names and isinstance(module, nn.Dropout):
                module.train()
                found.add(name)
        if not found:
            raise UserWarning(
                "No dropout layers found in model, maybe dropout is implemented "
                "via specialized layers?"
            )

    def _samples(self, X: Tensor) -> Tensor:
        """Return samples with a leading sample axis kept method-local."""
        self.activate_dropout()
        with torch.no_grad():
            samples = [self.model(X) for _ in range(self.num_mc_samples)]
        if not all(isinstance(sample, Tensor) for sample in samples):
            raise TypeError("Canonical MCDropout requires Tensor model outputs.")
        return torch.stack(samples)

    def _regression_payload(self, samples: Tensor) -> dict[str, Tensor]:
        """Convert sampled regression outputs without a task shape heuristic."""
        if self.prediction_kind == "gaussian":
            if samples.ndim < 3 or samples.shape[2] != 2:
                raise ValueError(
                    "Gaussian MCDropout requires an explicit two-channel output at axis 1."
                )
            means = samples[:, :, 0:1, ...]
            log_variances = samples[:, :, 1:2, ...]
            variances = torch.exp(log_variances).clamp_min(1e-6)
            prediction = means.mean(dim=0)
            epistemic = means.std(dim=0, correction=0)
            aleatoric = variances.mean(dim=0).sqrt()
            return {
                "pred": prediction,
                "pred_uct": (epistemic.square() + aleatoric.square()).sqrt(),
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
            }
        prediction = samples.mean(dim=0)
        epistemic = samples.std(dim=0, correction=0)
        return {"pred": prediction, "pred_uct": epistemic, "epistemic_uct": epistemic}

    def metric_prediction(self, raw_output: Tensor, stage: str) -> Tensor:
        """Select the explicit Gaussian mean for regression metrics.

        Point predictions retain every target channel. A Gaussian prediction
        has method-owned ``[mean, log_variance]`` channels, so only its mean is
        a valid target-shaped metric input.
        """
        if (
            not isinstance(self.task, ClassificationTask)
            and self.prediction_kind == "gaussian"
        ):
            if raw_output.ndim < 2 or raw_output.shape[1] != 2:
                raise ValueError(
                    "Gaussian MCDropout requires an explicit two-channel output at axis 1."
                )
            return raw_output[:, 0:1, ...]
        return super().metric_prediction(raw_output, stage)

    def prediction_payload(self, raw_output: Tensor) -> dict[str, Tensor]:
        """Convert canonical MC samples to the declared public payload."""
        if not isinstance(self.task, ClassificationTask):
            return self._regression_payload(raw_output)

        # ``raw_output`` is [sample, batch, class, ...].  The task determines
        # the distribution; no output-size inference is used here.
        mean_logits = raw_output.mean(dim=0)
        self._validate_raw_output(mean_logits)
        if self.task.mode == "multilabel":
            probabilities = torch.sigmoid(raw_output).mean(dim=0)
            clipped = probabilities.clamp(1e-7, 1 - 1e-7)
            entropy = -(
                clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()
            ).sum(dim=1)
        elif self.task.mode == "binary" and self.task.binary_encoding == "one_logit":
            probabilities = torch.sigmoid(raw_output).mean(dim=0)
            clipped = probabilities.clamp(1e-7, 1 - 1e-7)
            entropy = -(
                clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()
            ).select(1, 0)
        else:
            probabilities = torch.softmax(raw_output, dim=2).mean(dim=0).clamp_min(1e-7)
            entropy = -(probabilities * probabilities.log()).sum(dim=1)
        return {
            "pred": probabilities,
            "pred_uct": entropy,
            "logits": raw_output.movedim(0, -1),
        }

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Produce an MC-aggregated, contract-checked prediction payload."""
        del batch_idx, dataloader_idx
        payload = self.prediction_payload(self._samples(X))
        self.output_schema.validate_payload(payload)
        return payload

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Train with method-owned burn-in while retaining canonical metrics."""
        if self.current_epoch >= self.burnin_epochs or isinstance(
            self.task, ClassificationTask
        ):
            return super().training_step(batch, batch_idx, dataloader_idx)
        raw_output = self.forward(batch[self.input_key])
        if not isinstance(raw_output, Tensor):
            raise TypeError("Canonical MCDropout requires a Tensor model output.")
        metric_output = self.metric_prediction(raw_output, "train")
        target = self.task_runtime.target_for_loss(batch[self.target_key], raw_output)
        loss = nn.functional.mse_loss(metric_output, target)
        self.task_runtime.update_metrics("train", metric_output, batch[self.target_key])
        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Test from one canonical MC sample stack rather than two forwards."""
        del batch_idx, dataloader_idx
        samples = self._samples(batch[self.input_key])
        payload = self.prediction_payload(samples)
        self.output_schema.validate_payload(payload)
        self.task_runtime.update_metrics(
            "test",
            self.metric_prediction(samples.mean(dim=0), "test"),
            batch[self.target_key],
        )
        return self.task_runtime.test_result(
            payload, batch, input_key=self.input_key, target_key=self.target_key
        )
