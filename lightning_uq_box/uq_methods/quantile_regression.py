"""Implement Quantile Regression Model."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_sample_mean_std_from_quantile

from .base import DeterministicModel
from .loss_functions import HuberQLoss, QuantileLoss
from .utils import _get_num_outputs, default_regression_metrics


class QuantileRegressionBase(DeterministicModel):
    """Quantile Regression Base Module.

    If you use this model in your research, please cite the following paper:

    * https://www.jstor.org/stable/1913643
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of Quantile Regression Model.

        Args:
            model: pytorch model
            loss_fn: loss function
            quantiles: quantiles to compute
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
        """
        assert all(i < 1 for i in quantiles), "Quantiles should be less than 1."
        assert all(i > 0 for i in quantiles), "Quantiles should be greater than 0."
        assert _get_num_outputs(model) == len(
            quantiles
        ), "The number of desired quantiles should match the number of outputs of the model."

        super().__init__(model, loss_fn, optimizer, lr_scheduler)

        if loss_fn is None:
            self.loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        self.quantiles = quantiles
        self.median_index = self.quantiles.index(0.5)

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")


class QuantileRegression(QuantileRegressionBase):
    """Quantile Regression Module for Regression.

    If you use this model in your research, please cite the following paper:

    * https://www.jstor.org/stable/1913643
    """

    # def __init__(
    #     self,
    #     model: nn.Module,
    #     loss_fn: Optional[nn.Module] = None,
    #     quantiles: list[float] = [0.1, 0.5, 0.9],
    #     optimizer: OptimizerCallable = torch.optim.Adam,
    #     lr_scheduler: LRSchedulerCallable = None,
    # ) -> None:
    #     """Initialize a new instance of Quantile Regression Model.

    #     Args:
    #         model: pytorch model
    #         optimizer: optimizer used for training
    #         loss_fn: loss function
    #         lr_scheduler: learning rate scheduler
    #         quantiles: quantiles to compute
    #     """
    #     super().__init__(model, loss_fn, quantiles, optimizer, lr_scheduler)
    #     self.save_hyperparameters(
    #         ignore=["model", "loss_fn", "optimizer", "lr_scheduler"]
    #     )

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean/median prediction from quantile regression model.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, self.median_index : self.median_index + 1]  # noqa: E203

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Predict step with Quantile Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            predicted uncertainties
        """
        with torch.no_grad():
            out = self.model(X)  # [batch_size, len(self.quantiles)]
            np_out = out.cpu().numpy()

        median = self.extract_mean_output(out)
        mean, std = compute_sample_mean_std_from_quantile(
            np_out, self.hparams.quantiles
        )

        # TODO can happen due to overlapping quantiles
        # how to handle this properly ?
        std[std <= 0] = 1e-2

        return {
            "pred": median,
            "pred_uct": std,
            "lower_quant": np_out[:, 0],
            "upper_quant": np_out[:, -1],
            "aleatoric_uct": std,
        }


# class QuantilePxRegression(QuantileRegressionBase):
#     """Quantile Regression for Pixelwise Regression."""

#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: OptimizerCallable = torch.optim.Adam,
#         loss_fn: nn.Module = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
#         lr_scheduler: LRSchedulerCallable = None,
#
#     ) -> None:
#         super().__init__(model, optimizer, loss_fn, lr_scheduler)
#         self.save_hyperparameters(ignore=["model", "loss_fn"])
