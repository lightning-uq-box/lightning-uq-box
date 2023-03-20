"""Deep Kernel Learning."""

import os
from typing import Any, Dict

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.models._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .utils import save_predictions_to_csv


class DeepKernelLearningModel(gpytorch.Module, LightningModule):
    """Deep Kernel Learning Model.

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/1511.02222
    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        backbone: type[nn.Module],
        backbone_args: Dict[str, Any],
        gp: type[ApproximateGP],
        gp_args: Dict[str, Any],
        elbo_fn: type[_ApproximateMarginalLogLikelihood],
        num_train_points: int,
        lr: float,
        loss_fn: str,
        save_dir: str,
    ) -> None:
        """Initialize a new Deep Kernel Learning Model.

        Args:
            backbone: feature extractor class
            backbone_args: arguments to initialize the backbone
            gp: gpytorch module that takes extracted features as inputs
            gp_args: arguments to initializ the gp
            elbo_fn: gpytorch elbo functions
            num_train_points: number of training points necessary f
                or Gpytorch elbo function
            lr:
            loss_fn:
            save_dir:
        """
        super().__init__()
        self.save_hyperparameters()

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
        self.backbone = self.hparams.backbone(**self.hparams.backbone_args)
        # rm last linear layer if available

        self.gp = self.hparams.gp(**self.hparams.gp_args)

        self.elbo_fn = VariationalELBO(
            GaussianLikelihood(), self.gp, num_data=self.hparams.num_data
        )

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        """Forward pass through model.

        Args:
            X: input tensor to backbone

        Returns:
            output from GP
        """
        return self.gp(self.backbone(X))

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Training step for DKL model."""
        X, y = args[0]

        y_pred = self.forward(X)
        loss = -self.elbo_fn(y_pred, y)

        self.log("val_loss", loss)  # logging to Logger
        self.train_metrics(y_pred, y)

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
        loss = -self.elbo_fn(y_pred, y)

        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(y_pred, y)

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
        # TODO create prediction dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers # noqa: E501
        """
        # in the paper they use SGD? Does it matter?
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}


class GP(ApproximateGP):
    """Gaussian Process Model.

    Taken from https://github.com/y0ast/DUE/blob/f29c990811fd6a8e76215f17049e6952ef5ea0c9/due/dkl.py#L62 # noqa: E501
    """

    kernel_choices = ["RBF", "Matern12", "Matern32", "Matern52", "RQ"]

    def __init__(
        self,
        num_outputs: int,
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

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
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
