# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

# adapted from https://github.com/y0ast/DUE/blob/main/due/sngp.py

"""Spectral Normalized Gaussian Process (SNGP)."""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.adam import Adam as Adam

from .base import BaseModule
from .spectral_normalized_layers import (
    collect_input_sizes,
    spectral_normalize_model_layers,
)
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    save_classification_predictions,
    save_regression_predictions,
)

# TODO
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/understanding/sngp.ipynb
# https://www.tensorflow.org/tutorials/understanding/sngp
# good visualizations and explanations here


class SNGPBase(BaseModule):
    """Specral Normalized Gaussian Process (SNGP)."""

    pred_file_name = "preds.csv"

    def __init__(
        self,
        feature_extractor: nn.Module,
        loss_fn: nn.Module,
        num_targets: int = 1,
        num_gp_features: int = 128,
        num_random_features: int = 1024,
        normalize_gp_features: bool = True,
        feature_scale: int = 2,
        ridge_penalty: float = 1.0,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        input_size: Optional[int] = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new SNGP model.

        Args:
            feature_extractor: Feature extractor model
            loss_fn: Loss function
            num_targets: Number of output units / targets
            num_gp_features: Number of GP features
            num_deep_features: Number of deep features
            num_random_features: Number of random features
            normalize_gp_features: Whether to normalize GP features
            feature_scale: Feature scale
            ridge_penalty: Ridge penalty
            coeff: soft normalization only when sigma larger than coeff,
                should be (0, 1)
            n_power_iterations: number of power iterations for spectral normalization
            input_size: image dimension input size needed for spectral normalization
            freeze_backbone: whether to freeze the feature extractor
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
        """
        super().__init__()

        # spectral normalize feature extractor
        self.input_size = input_size
        self.input_dimensions = collect_input_sizes(feature_extractor, self.input_size)
        feature_extractor = spectral_normalize_model_layers(
            feature_extractor, n_power_iterations, self.input_dimensions, coeff
        )
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.num_targets = num_targets
        self.num_gp_features = num_gp_features
        # number of output features from feature extractor
        self.num_deep_features = _get_num_outputs(feature_extractor)
        self.num_random_features = num_random_features
        self.normalize_gp_features = normalize_gp_features
        self.feature_scale = feature_scale
        self.ridge_penalty = ridge_penalty

        self.freeze_backbone = freeze_backbone

        self._build_model()

        self.setup_task()

    def _build_model(self) -> None:
        """Build SNGP model."""
        if self.num_gp_features > 0:
            self.num_gp_features = self.num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (self.num_gp_features, self.num_deep_features)),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = self.num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = self.normalize_gp_features
        if self.normalize_gp_features:
            self.normalize = nn.LayerNorm(self.num_gp_features)

        self.rff = RandomFourierFeatures(
            self.num_gp_features, self.num_random_features, self.feature_scale
        )
        self.beta = nn.Linear(self.num_random_features, self.num_targets)

        self.register_buffer("seen_data", torch.tensor(0))

        precision = torch.eye(self.num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(self.num_random_features))

        # freeze feature extractor
        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor]:
        """Forward pass of the SNGP model.

        Args:
            x: Input tensor

        Returns:
            Output tensor after applying SNGP
        """
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)
        k = self.rff(f_reduc)
        return self.beta(k), k

    def reset_precision_matrix(self):
        """Reset the precision matrix to identity matrix."""
        identity = torch.eye(self.precision.shape[0], device=self.precision.device)
        self.precision = identity * self.ridge_penalty
        self.seen_data = torch.tensor(0)
        self.recompute_covariance = True

    def recompute_covariance_matrix(self):
        """Recompute the covariance matrix."""
        with torch.no_grad():
            eps = 1e-7
            jitter = eps * torch.eye(
                self.precision.shape[1], device=self.precision.device
            )
            u, info = torch.linalg.cholesky_ex(self.precision + jitter)
            assert (info == 0).all(), "Precision matrix inversion failed!"
            torch.cholesky_inverse(u, out=self.covariance)

    def on_fit_start(self) -> None:
        """Before fitting compute number of training points."""
        self.num_data = len(self.trainer.datamodule.train_dataloader().dataset)

    def on_train_epoch_start(self) -> None:
        """Called when the train epoch begins."""
        self.reset_precision_matrix()

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
        pred, k = self.forward(batch[self.input_key])
        precision_minibatch = k.t() @ k
        self.precision += precision_minibatch

        loss = self.loss_fn(pred, batch[self.target_key])
        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.target_key].shape[0] > 1:
            self.train_metrics(pred, batch[self.target_key])

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        pred_dict = self.predict_step(batch[self.input_key])

        loss = self.loss_fn(pred_dict["pred"], batch[self.target_key])
        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.target_key].shape[0] > 1:
            self.val_metrics(pred_dict["pred"], batch[self.target_key])

        return loss

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        if self.trainer.current_epoch % 2 == 0:
            self.recompute_covariance_matrix()
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            predictions
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key]
        loss = self.loss_fn(pred_dict["pred"], batch[self.target_key])
        self.log("test_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.target_key].shape[0] > 1:
            self.test_metrics(pred_dict["pred"], batch[self.target_key])

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)
        del pred_dict["pred_cov"]
        return pred_dict

    def on_test_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, X: Tensor) -> dict[str, Tensor]:
        """Predict the output for a batch of inputs.

        Args:
            X: Input tensor

        Returns:
            The predicted output
        """
        with torch.no_grad():
            pred, k = self.forward(X)
            pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

        output_std = pred_cov.diag().sqrt()

        return {
            "pred": pred,
            "pred_uct": output_std,
            "epistemic_uct": output_std,
            "pred_cov": pred_cov,
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.optimizer(
            [
                {"params": self.feature_extractor.parameters()},
                {"params": self.beta.parameters()},
            ]
        )
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class SNGPRegression(SNGPBase):
    """SNGP for regression."""

    def setup_task(self) -> None:
        """Set up task."""
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


class SNGPClassification(SNGPBase):
    """SNGP for classification."""

    valid_tasks = ["binary", "multiclass"]

    def __init__(
        self,
        feature_extractor: Module,
        loss_fn: Module,
        num_targets: int = 1,
        num_gp_features: int = 128,
        num_random_features: int = 1024,
        normalize_gp_features: bool = True,
        feature_scale: int = 2,
        ridge_penalty: float = 1,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        input_size: Optional[int] = None,
        mean_field_factor: Optional[float] = math.pi / 8,
        task: str = "multiclass",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new SNGP model for classification.

        Args:
            feature_extractor: Feature extractor model
            loss_fn: Loss function
            num_targets: Number of output units / targets
            num_gp_features: Number of GP features
            num_deep_features: Number of deep features
            num_random_features: Number of random features
            normalize_gp_features: Whether to normalize GP features
            feature_scale: Feature scale
            ridge_penalty: Ridge penalty
            coeff: soft normalization only when sigma larger than coeff,
                should be (0, 1)
            n_power_iterations: number of power iterations for spectral normalization
            input_size: image dimension input size needed for spectral normalization
            mean_field_factor: Mean field factor, required for classification problems
            task: classification task, one of ['binary', 'multiclass']
            freeze_backbone: whether to freeze the feature extractor
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
        """
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        self.mean_field_factor = mean_field_factor
        super().__init__(
            feature_extractor,
            loss_fn,
            num_targets,
            num_gp_features,
            num_random_features,
            normalize_gp_features,
            feature_scale,
            ridge_penalty,
            coeff,
            n_power_iterations,
            input_size,
            freeze_backbone,
            optimizer,
            lr_scheduler,
        )

    def setup_task(self) -> None:
        """Set up task."""
        self.train_metrics = default_classification_metrics(
            "train", task=self.task, num_classes=self.num_targets
        )
        self.val_metrics = default_classification_metrics(
            "val", task=self.task, num_classes=self.num_targets
        )
        self.test_metrics = default_classification_metrics(
            "test", task=self.task, num_classes=self.num_targets
        )

    def mean_field_logits(self, logits: Tensor, pred_cov: Tensor) -> Tensor:
        """Applies the Mean-Field approximation to the provided logits.

        Based on: https://arxiv.org/abs/2006.07584

        Args:
            logits: The logits to be transformed
            pred_cov: The predicted covariance

        Returns:
            The transformed logits
        """
        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

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

    def predict_step(self, X: Tensor) -> dict[str, Tensor]:
        """Predict the output for a batch of inputs.

        Args:
            X: Input tensor

        Returns:
            output dictionary
        """
        pred_dict = super().predict_step(X)
        pred_dict["pred"] = self.mean_field_logits(
            pred_dict["pred"], pred_dict["pred_cov"]
        )
        pred_dict["logits"] = pred_dict["pred"]
        return pred_dict


def random_ortho(n: int, m: int) -> Tensor:
    """Generate a random orthogonal matrix.

    Args:
        n: Number of rows in the output matrix
        m: Number of columns in the output matrix

    Returns:
        A random orthogonal matrix of size (n, m)
    """
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for Gaussian Processes."""

    def __init__(
        self,
        in_dim: int,
        num_random_features: int,
        feature_scale: Optional[float] = None,
    ):
        """Initialize a new instance Random Fourier Features for GP.

        Args:
            in_dim: Input dimension
            num_random_features: Number of random features
            feature_scale: Feature scale. If None,
                it is set to sqrt(num_random_features / 2)
        """
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)

        self.register_buffer("feature_scale", torch.tensor(feature_scale))

        if num_random_features <= in_dim:
            W = random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)

        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Random Fourier Features module.

        Args:
            x: Input tensor

        Returns:
            Output tensor after applying Random Fourier Features
        """
        k = torch.cos(x @ self.W + self.b)
        k = k / self.feature_scale

        return k
