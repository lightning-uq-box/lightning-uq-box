# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Deep Kernel Learning."""


import os
from typing import Any, Dict

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from sklearn import cluster
from torch import Tensor
from torch.utils.data import Dataset

from .base import BaseModule
from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    save_classification_predictions,
    save_regression_predictions,
)


class DKLBase(gpytorch.Module, BaseModule):
    """Deep Kernel Learning Base Module.

    If you use this model in your work, please cite:

    * https://proceedings.mlr.press/v51/wilson16.html
    """

    # TODO make elbo_fn an argument that can be instatiated with
    # different elbo functions and Lightning CLI
    kernel_choices = ["RBF", "Matern12", "Matern32", "Matern52", "RQ"]
    pred_file_name = "preds.csv"

    def __init__(
        self,
        feature_extractor: nn.Module,
        n_inducing_points: int,
        gp_kernel: str = "RBF",
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deep Kernel Learning Model.

        Initialize a new Deep Kernel Learning Model.

        Args:
            feature_extractor: feature extractor model
            n_inducing_points: number of inducing points
            gp_kernel: kernel choice, supports one of
                ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']
            elbo_fn: gpytorch elbo function used for optimization
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "feature_extractor",
                "gp_layer",
                "optimizer",
                "lr_scheduler",
                "elbo_fn",
            ]
        )

        assert (
            gp_kernel in self.kernel_choices
        ), "Please choose one of the supported kernel choices ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']"  # noqa: E501
        self.gp_kernel = gp_kernel
        self.optimizer = optimizer
        self.feature_extractor = feature_extractor

        self.lr_scheduler = lr_scheduler

        self.dkl_model_built = False

        self.setup_task()

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.feature_extractor)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension from model
        """
        return self.gp_layer.n_outputs

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        raise NotImplementedError

    def _fit_initial_lengthscale_and_inducing_points(self) -> None:
        """Fit the initial lengthscale and inducing points for DKL."""
        train_dataset = self.trainer.datamodule.train_dataloader().dataset

        def augmentation(batch: dict[str, torch.Tensor]):
            """Gather augmentations from datamodule."""
            # apply datamodule augmentation
            aug_batch = self.trainer.datamodule.on_after_batch_transfer(
                batch, dataloader_idx=0
            )
            return aug_batch[self.input_key]

        self.n_train_points = len(train_dataset)
        self.initial_inducing_points, self.initial_lengthscale = compute_initial_values(
            train_dataset,
            self.feature_extractor,
            self.hparams.n_inducing_points,
            augmentation,
            self.input_key,
            self.target_key,
        )
        self.initial_inducing_points = self.initial_inducing_points.to(self.device)
        self.initial_lengthscale = self.initial_lengthscale.to(self.device)

        # build the model ready for training
        self._build_model()

        self.dkl_model_built = True

    def forward(self, X: Tensor, **kwargs) -> MultivariateNormal:
        """Forward pass through model.

        Args:
            X: input tensor to backbone

        Returns:
            output from GP
        """
        features = self.feature_extractor(X)
        scaled_features = self.scale_to_bounds(features)
        output = self.gp_layer(scaled_features)
        return output

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        y_pred = self.forward(X)
        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(y_pred.mean, y.squeeze(-1))
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

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        # in sanity checking GPPYtorch is not in eval
        # and we get a device error
        if self.trainer.sanity_checking:
            self.scale_to_bounds.train()
            y_pred = self.forward(X)
            self.scale_to_bounds.eval()
        else:
            y_pred = self.forward(X)

        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log(
            "val_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger

        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(y_pred.mean, y.squeeze(-1))
        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step."""
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()

        self.log(
            "test_loss",
            -self.elbo_fn(out_dict["out"], batch[self.target_key].squeeze(-1)),
            batch_size=batch[self.input_key].shape[0],
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key].squeeze(-1))

        del out_dict["out"]

        out_dict["pred"] = out_dict["pred"].detach().cpu()

        # save metadata
        out_dict = self.add_aux_data_to_dict(out_dict, batch)
        return out_dict

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        # need to create models here given the order of hooks
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        if not self.dkl_model_built:
            self._fit_initial_lengthscale_and_inducing_points()

        optimizer = self.optimizer(
            [
                {"params": self.feature_extractor.parameters()},
                {"params": self.gp_layer.hyperparameters()},
                {"params": self.gp_layer.variational_parameters()},
                {"params": self.likelihood.parameters()},
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


class DKLRegression(DKLBase):
    """Deep Kernel Learning Model.

    If you use this model in your research, please cite the following papers:

    * https://proceedings.mlr.press/v51/wilson16.html
    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        n_inducing_points: int,
        num_targets: int = 1,
        gp_kernel: str = "RBF",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deep Kernel Learning Model for Regression.

        Args:
            feature_extractor: feature extractor model
            n_inducing_points: number of inducing points
            num_targets: number of targets
            gp_kernel: kernel choice, supports one of
                ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']
            elbo_fn: gpytorch elbo function used for optimization
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        self.freeze_backbone = freeze_backbone

        super().__init__(
            feature_extractor, n_inducing_points, gp_kernel, optimizer, lr_scheduler
        )

        self.save_hyperparameters(
            ignore=[
                "feature_extractor",
                "gp_layer",
                "optimizer",
                "lr_scheduler",
                "elbo_fn",
            ]
        )
        self.num_targets = num_targets

        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def _build_model(self) -> None:
        """Build the model ready for training."""
        self.gp_layer = DKLGPLayer(
            n_outputs=self.num_targets,
            initial_lengthscale=self.initial_lengthscale,
            initial_inducing_points=self.initial_inducing_points,
            kernel=self.gp_kernel,
        )
        self.scale_to_bounds = ScaleToBounds(-2.0, 2.0)
        self.likelihood = GaussianLikelihood()

        self.elbo_fn = VariationalELBO(
            self.likelihood, self.gp_layer, num_data=self.n_train_points
        )

        # put gpytorch modules on cuda
        if self.device.type == "cuda":
            self.gp_layer = self.gp_layer.cuda()
            self.likelihood = self.likelihood.cuda()

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
        if not self.dkl_model_built:
            self._fit_initial_lengthscale_and_inducing_points()
        self.feature_extractor.eval()
        self.gp_layer.eval()
        self.likelihood.eval()

        # TODO make num samples an argument
        with (
            torch.no_grad(),
            gpytorch.settings.num_likelihood_samples(64),
            gpytorch.settings.fast_pred_var(state=False),
        ):
            output = self.likelihood(self.forward(X))
            mean = output.mean
            std = output.stddev.cpu()

        return {"pred": mean, "pred_uct": std, "epistemic_uct": std, "out": output}


class DKLClassification(DKLBase):
    """Deep Kernel Learning for Classification.

    If you use this model in your research, please cite the following papers:

    * https://proceedings.mlr.press/v51/wilson16.html
    * https://arxiv.org/abs/2102.11409
    """

    valid_tasks = ["binary", "multiclass", "multilable"]

    # TODO
    # gp_layer: Callable[[only. the two args that are needed from computation],
    # DKLGPLayer]
    # similar to optimizer only include the arguments in the callable args section
    # that are missing from conf file
    def __init__(
        self,
        feature_extractor: nn.Module,
        n_inducing_points: int,
        num_classes: int,
        task: str = "multiclass",
        gp_kernel: str = "RBF",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new Deep Kernel Learning Model for Classification.

        Args:
            feature_extractor: feature extractor model
            n_inducing_points: number of inducing points
            gp_kernel: GP kernel choice, supports one of
                'RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']
            num_classes: number of classes
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            freeze_backbone: whether to freeze the backbone
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        assert task in self.valid_tasks
        self.task = task

        self.num_classes = num_classes
        # number of latent features of the feature extractor
        self.num_features = _get_num_outputs(feature_extractor)
        self.freeze_backbone = freeze_backbone

        super().__init__(
            feature_extractor, n_inducing_points, gp_kernel, optimizer, lr_scheduler
        )

        self.save_hyperparameters(
            ignore=[
                "feature_extractor",
                "gp_layer",
                "optimizer",
                "lr_scheduler",
                "elbo_fn",
            ]
        )

        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

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

    def _adapt_output_for_metrics(self, output: MultivariateNormal) -> Tensor:
        """Adapt model output to be compatible for metric computation.."""
        return output.mean

    def _build_model(self) -> None:
        """Build the model ready for training."""
        self.gp_layer = DKLGPLayer(
            n_outputs=self.num_features,
            initial_lengthscale=self.initial_lengthscale,
            initial_inducing_points=self.initial_inducing_points,
            kernel=self.gp_kernel,
        )
        self.scale_to_bounds = ScaleToBounds(-2.0, 2.0)
        self.likelihood = SoftmaxLikelihood(
            num_classes=self.num_classes, num_features=self.num_features
        )
        self.elbo_fn = VariationalELBO(
            self.likelihood, self.gp_layer, num_data=self.n_train_points
        )

        # put gpytorch modules on cuda
        if self.device.type == "cuda":
            self.gp_layer = self.gp_layer.cuda()
            self.likelihood = self.likelihood.cuda()

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        y_pred = self.forward(X)
        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        scores = self.likelihood(y_pred).probs.mean(0)
        self.train_metrics(scores, y.squeeze(-1))
        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]

        # in sanity checking GPPYtorch is not in eval
        # and we get a device error
        if self.trainer.sanity_checking:
            self.scale_to_bounds.train()
            y_pred = self.forward(X)
            self.scale_to_bounds.eval()
        else:
            y_pred = self.forward(X)

        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log(
            "val_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        scores = self.likelihood(y_pred).probs.mean(0)
        self.val_metrics(scores, y.squeeze(-1))
        return loss

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
        if not self.dkl_model_built:
            self._fit_initial_lengthscale_and_inducing_points()
        self.feature_extractor.eval()
        self.gp_layer.eval()
        self.likelihood.eval()

        # TODO make num samples an argument
        with (
            torch.no_grad(),
            gpytorch.settings.num_likelihood_samples(64),
            gpytorch.settings.fast_pred_var(state=False),
        ):
            gp_dist = self.forward(X)
            output = self.likelihood(gp_dist)
            mean = output.probs.mean(0)  # take mean over sampling dimension
            entropy = output.entropy().mean(0).cpu()
        return {"pred": mean, "pred_uct": entropy, "out": gp_dist, "logits": mean}

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


class DKLGPLayer(ApproximateGP):
    """Gaussian Process Model for Deep Kernel Learning.

    Taken from https://github.com/y0ast/DUE/blob/f29c990811fd6a8e76215f17049e6952ef5ea0c9/due/dkl.py#L62 # noqa: E501
    """

    kernel_choices = ["RBF", "Matern12", "Matern32", "Matern52", "RQ"]

    def __init__(
        self,
        n_outputs: int,
        initial_lengthscale: Tensor,
        initial_inducing_points: Tensor,
        kernel: str = "Matern32",
    ):
        """Initialize a new instance of the Gaussian Process Layer.

        Args:
            n_outpus: number of latent output features of the GP
            initial_lengthscale: initial lengthscale to use
            initial_inducing_points: initial inducing points to use
            kernel: kernel choice, supports one of
                ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']

        Raises:
            ValueError if kernel is not supported
        """
        n_inducing_points = initial_inducing_points.shape[0]

        self.n_outputs = n_outputs

        if n_outputs > 1:
            batch_shape = torch.Size([n_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if n_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=n_outputs
            )

        super().__init__(variational_strategy)

        kwargs = {"batch_shape": batch_shape}

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError(
                "Specified kernel not known, supported kernel "
                f"choices are {self.kernel_choices}."
            )

        kernel.lengthscale = initial_lengthscale.cpu() * torch.ones_like(
            kernel.lengthscale
        )

        self.mean_module = ConstantMean(batch_shape=batch_shape)

        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Forward pass of GP.

        Args:
            X: input to GP of shape [batch_size, num_input_features]
        """
        mean = self.mean_module(X)
        covar = self.covar_module(X)
        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        """Return inducing points."""
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return


def compute_initial_values(
    train_dataset: Dataset,
    feature_extractor: nn.Module,
    n_inducing_points: int,
    augmentation,
    input_key: str,
    target_key: str,
) -> tuple[Tensor]:
    """Compute the inital values.

    Args:
        train_dataset: training dataset to compute the initial values on
        feature_extractor: feature extractor with which to compute the initial values
        n_inducing_points: number of inducing points
        augmentation: augmentation function applied to the dataset samples
        input_key: input key of dictionary that gets returned by dataset
        target_key: target key of dictionary that gets returned by dataset

    Returns:
        initial inducing points and initial lengthscale
    """
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO find a universal solution for the dataset key
    with torch.no_grad():
        feature_extractor = feature_extractor.to(device)
        for i in range(steps):
            random_indices = idx[i].tolist()

            if isinstance(train_dataset[0], dict):
                X_sample = torch.stack(
                    [train_dataset[j][input_key] for j in random_indices]
                )
                y_sample = torch.stack(
                    [train_dataset[j][target_key] for j in random_indices]
                )
            else:
                X_sample = torch.stack([train_dataset[j][0] for j in random_indices])
                y_sample = torch.stack([train_dataset[j][1] for j in random_indices])

            X_sample = X_sample.to(device)
            y_sample = y_sample.to(device)

            X_sample = augmentation({input_key: X_sample, target_key: y_sample})
            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)
    return initial_inducing_points.to(torch.float), initial_lengthscale.to(torch.float)


def _get_initial_inducing_points(
    f_X_sample: np.ndarray, n_inducing_points: int
) -> Tensor:
    """Compute the initial number of inducing points.

    Args:
        f_X_sample: feature extractor output samples
        n_inducing_points: number of inducing points

    Returns:
        initial inducing points
    """
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10, n_init=3
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples: Tensor) -> Tensor:
    """Compute the initial lengthscale.

    Args:
        f_X_samples: feature extractor output samples

    Returns:
        length scale tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_X_samples = f_X_samples.to(device)

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()
