# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

# adapted from https://github.com/y0ast/DUE/blob/main/due/sngp.py

"""Spectral Normalized Gaussian Process (SNGP)."""

import math
from typing import Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import BaseModule
from .utils import default_classification_metrics, default_regression_metrics


class SNGP(BaseModule):
    """Specral Normalized Gaussian Process (SNGP)."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        loss_fn: nn.Module,
        num_data: int,
        num_gp_features: int = 128,
        num_random_features: int = 1024,
        normalize_gp_features: int = True,
        feature_scale: int = 2,
        ridge_penalty: float = 1.0,
        mean_field_factor: Optional[float] = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new SNGP model.

        Args:
            feature_extractor: Feature extractor model
            loss_fn: Loss function
            num_data: Number of data points
            num_gp_features: Number of GP features
            num_random_features: Number of random features
            normalize_gp_features: Whether to normalize GP features
            feature_scale: Feature scale
            ridge_penalty: Ridge penalty
            mean_field_factor: Mean field factor
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # TODO figure this out from dataset and trainer
        self.num_data = num_data
        self.num_gp_features = num_gp_features
        self.num_random_features = num_random_features
        self.normalize_gp_features = normalize_gp_features
        self.feature_scale = feature_scale
        self.ridge_penalty = ridge_penalty
        self.mean_field_factor = mean_field_factor

        self._build_model()

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
        self.beta = nn.Linear(self.num_random_features, self.num_outputs)

        self.num_data = self.num_data
        self.register_buffer("seen_data", torch.tensor(0))

        precision = torch.eye(self.num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(self.num_random_features))

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
        y = batch[self.target_key]
        pred, k = self.forward(batch[self.input_key])
        # precision_minibatch = k.t() @ k
        # self.precision += precision_minibatch
        # self.seen_data += x.shape[0]

        # assert (
        #     self.seen_data <= self.num_data
        # ), "Did not reset precision matrix at start of epoch"
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        if y.shape[0] > 1:
            self.train_metrics(pred, y)

        return loss

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
        pass

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            predictions
        """
        pass

    def predict_step(self, X: Tensor) -> Tensor:
        """Predict the output for a batch of inputs.

        Args:
            X: Input tensor

        Returns:
            The predicted output
        """
        pred, k = self.forward(X)
        pass


class SNGPRegression(SNGP):
    """SNGP for regression."""

    def setup_task(self) -> None:
        """Set up task."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")


class SNGPClassification(SNGP):
    """SNGP for classification."""

    def setup_task(self) -> None:
        """Set up task."""
        self.train_metrics = default_classification_metrics(
            "train", task=self.task, num_classes=self.num_outputs
        )
        self.val_metrics = default_classification_metrics(
            "val", task=self.task, num_classes=self.num_outputs
        )
        self.test_metrics = default_classification_metrics(
            "test", task=self.task, num_classes=self.num_outputs
        )


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


class Laplace(nn.Module):
    """Laplace approximation for Gaussian Processes."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_deep_features: int,
        num_gp_features: int,
        normalize_gp_features: bool,
        num_random_features: int,
        num_outputs: int,
        num_data: int,
        train_batch_size: int,
        ridge_penalty: float = 1.0,
        feature_scale: Optional[float] = None,
        mean_field_factor: Optional[float] = None,
    ) -> None:
        """Initializes the Laplace approximation for GP.

        Args:
            feature_extractor: The feature extractor network
            num_deep_features: Number of deep features
            num_gp_features: Number of Gaussian Process features
            normalize_gp_features: Whether to normalize Gaussian Process features
            num_random_features: Number of random features
            num_outputs: Number of output units
            num_data: Number of data points
            train_batch_size: Training batch size
            ridge_penalty: Ridge penalty. Defaults to 1.0
            feature_scale: Feature scale. Defaults to None
            mean_field_factor: Mean field factor, required for classification problems
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor
        self.ridge_penalty = ridge_penalty
        self.train_batch_size = train_batch_size

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = nn.LayerNorm(num_gp_features)

        self.rff = RandomFourierFeatures(
            num_gp_features, num_random_features, feature_scale
        )
        self.beta = nn.Linear(num_random_features, num_outputs)

        self.num_data = num_data
        self.register_buffer("seen_data", torch.tensor(0))

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(num_random_features))

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the model.

        Args:
            x: The input tensor.

        Returns:
            The model's output.
        """
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)

        k = self.rff(f_reduc)

        pred = self.beta(k)

        if self.training:
            precision_minibatch = k.t() @ k
            self.precision += precision_minibatch
            self.seen_data += x.shape[0]

            assert (
                self.seen_data <= self.num_data
            ), "Did not reset precision matrix at start of epoch"
        else:
            assert self.seen_data > (
                self.num_data - self.train_batch_size
            ), "Not seen sufficient data for precision matrix"

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision.shape[1], device=self.precision.device
                    )
                    u, info = torch.linalg.cholesky_ex(self.precision + jitter)
                    assert (info == 0).all(), "Precision matrix inversion failed!"
                    torch.cholesky_inverse(u, out=self.covariance)

                self.recompute_covariance = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov)

        return pred
