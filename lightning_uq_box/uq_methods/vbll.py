# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted for Lightning from https://github.com/VectorInstitute/vbll

"""Variational Bayesian Last Layer (VBLL)."""

from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.nn.modules import Module

from .base import DeterministicRegression
from .utils import _get_output_layer_name_and_module


class VBLLRegression(DeterministicRegression):
    """Variational Bayesian Last Layer (VBLL) for Regression.

    If you use this model in your research, please cite the following paper:

    * https://openreview.net/forum?id=Sx7BIiPzys

    """

    def __init__(
        self,
        model: Module,
        regularization_weight,
        num_targets: int = 1,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 1e-2,
        dof: int = 1,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize the VBLL regression model.

        Args:
            model: The backbone model
            regularization_weight : regularization weight term in ELBO, and should be
                1 / (dataset size) by default. This term impacts the epistemic
                uncertainty estimate.
            num_targets : Number of targets
            parameterization : Parameterization of covariance matrix. One of
                ['dense','diagonal']
            prior_scale : prior covariance matrix scale
                Scale of prior covariance matrix
            wishart_scale : Scale of Wishart prior on noise covariance. This term
                has an impact on the aleatoric uncertainty estimate.
            dof : Degrees of freedom of Wishart prior on noise covariance
            freeze_backbone: If True, the backbone model will be frozen
                and only the VBBL layer will be trained
            optimizer: The optimizer to use for training
            lr_scheduler: The learning rate scheduler to use for training

        """
        super().__init__(model, None, freeze_backbone, optimizer, lr_scheduler)

        try:
            from vbll import Regression as VBLLReg  # noqa: F401
        except ImportError:
            raise ImportError(
                "You need to install the vbll package: 'pip install vbll'."
            )

        self.regularization_weight = regularization_weight
        self.parameterization = parameterization
        self.prior_scale = prior_scale
        self.wishart_scale = wishart_scale
        self.dof = dof

        _, last_module_backbone = _get_output_layer_name_and_module(self.model)

        self.model = nn.Sequential(
            self.model,
            VBLLReg(
                in_features=last_module_backbone.out_features,
                out_features=num_targets,
                regularization_weight=regularization_weight,
                parameterization=parameterization,
                prior_scale=prior_scale,
                wishart_scale=wishart_scale,
                dof=dof,
            ),
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for metrics.

        Args:
            out: the output from the VBLL module

        Returns:
            the mean prediction
        """
        return out.predictive.mean

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            training loss
        """
        out = self.model(batch[self.input_key])
        loss = out.train_loss_fn(batch[self.target_key])

        self.log(
            "train_loss", loss, batch_size=batch[self.input_key].shape[0]
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            validation loss
        """
        out = self.model(batch[self.input_key])
        loss = out.val_loss_fn(batch[self.target_key])

        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            test loss
        """
        pred_dict = self.predict_step(batch[self.input_key])

        test_loss = pred_dict["out"].val_loss_fn(batch[self.target_key])

        self.log("test_loss", test_loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(pred_dict["out"]), batch[self.target_key]
            )

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)
        # delete out from pred_dict
        del pred_dict["out"]
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step with VBLL model."""
        with torch.no_grad():
            pred = self.model(X)

        return {
            "pred": pred.predictive.mean,
            "pred_uct": torch.sqrt(pred.predictive.covariance).squeeze(-1),
            "out": pred,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Optimizers."""
        # exclude vbll parameters from weight decay
        optimizer = self.optimizer(
            [
                {"params": self.model[0].parameters()},
                {"params": self.model[-1].parameters(), "weight_decay": 0.0},
            ]
        )
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
