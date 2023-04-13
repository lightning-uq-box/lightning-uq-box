"""Deep Kernel Learning."""

# TO DO:
# Use wrapper gpytorch.kernels.GridInterpolationKernel(base_kernel, grid_size,
# num_dims=None, grid_bounds=None, active_dims=None)
# for all kernels to have scalable approximate GPs
# write choice of grid size
# based on training data with helper function
# gpytorch.utils.grid.choose_grid_size(train_inputs,
# ratio=1.0, kronecker_structure=True)

import os
from typing import Any, Dict, List

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from lightning import LightningModule
from sklearn import cluster
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from uq_method_box.eval_utils import compute_quantiles_from_std

from .utils import save_predictions_to_csv


class DeepKernelLearningModel(gpytorch.Module, LightningModule):
    """Deep Kernel Learning Model.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/1511.02222
    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        gp_layer: type[ApproximateGP],
        gp_args: Dict[str, Any],
        elbo_fn: type[_ApproximateMarginalLogLikelihood],
        n_train_points: int,
        lr: float,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new Deep Kernel Learning Model.

        Args:
            backbone: feature extractor class
            backbone_args: arguments to initialize the backbone
            gp_layer: gpytorch module that takes extracted features as inputs
            gp_args: arguments to initializ the gp_layer
            elbo_fn: gpytorch elbo functions
            n_train_points: number of training points necessary f
                or Gpytorch elbo function
        """
        super().__init__()
        self.save_hyperparameters(ignore="feature_extractor")
        self.n_train_points = n_train_points
        self.lr = lr
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="test_",
        )

        self._build_model()

    def _build_model(self) -> None:
        """Build the model."""
        self.gp_layer = self.hparams.gp_layer(**self.hparams.gp_args)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = ScaleToBounds(-2.0, 2.0)

        self.likelihood = GaussianLikelihood()
        self.elbo_fn = self.hparams.elbo_fn(
            self.likelihood, self.gp_layer, num_data=self.n_train_points
        )

    def forward(self, X: Tensor, **kwargs) -> Tensor:
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

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Training step for DKL model."""
        X, y = args[0]

        y_pred = self.forward(X)
        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(y_pred.loc, y.squeeze(-1))
        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        y_pred = self.forward(X)
        loss = -self.elbo_fn(y_pred, y.squeeze(-1)).mean()

        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(y_pred.loc, y.squeeze(-1))
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).cpu().numpy()
        return out_dict

    def on_test_batch_end(
        self,
        outputs: Dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        save_predictions_to_csv(
            outputs, os.path.join(self.hparams.save_dir, "predictions.csv")
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        self.feature_extractor.eval()
        self.gp_layer.eval()
        self.likelihood.eval()

        with torch.no_grad():
            out = self.forward(X)  # GPytorch MultivariatNormal Object

        mean = out.mean.cpu()
        # preds.confidence_region() Returns 2 standard deviations
        # above and below the mean. Return type (torch.Tensor, torch.Tensor)
        confidence = out.confidence_region()

        std = ((confidence[1] - confidence[0]) * 0.5).cpu()

        quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
        return {
            "mean": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
        """
        optimizer = torch.optim.Adam(
            [
                {"params": self.feature_extractor.parameters()},
                {"params": self.gp_layer.hyperparameters()},
                {"params": self.gp_layer.variational_parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.lr,
        )
        return {"optimizer": optimizer}


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
        kernel: str = "RBF",
    ):
        """Initialize a new instance of the Gaussian Process model.

        Args:
            num_outpus: number of target outputs
            initial_lengthscale:
            initial_inducing_points:
            kernel: kernel choice, supports one of
                ['RBF', 'Matern12', 'Matern32', 'Matern52', 'RQ']

        Raises:
            ValueError if kernel is not supported
        """
        n_inducing_points = initial_inducing_points.shape[0]

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

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)

        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """Forward pass of GP.

        Args:
            X: input to GP of shape [batch_size, num_input_dims]
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


def initial_values(train_dataset, feature_extractor, n_inducing_points):
    """Compute the inital values."""
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    """Compute the initial number of inducing points."""
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    """Compute the initial lengthscale."""
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()
