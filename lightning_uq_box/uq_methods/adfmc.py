# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Assumed Density Filtering with/with out MC Dropout."""

import copy
import math
import os
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from lightning_uq_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)
from lightning_uq_box.models.adf_layers.adf_layers import Softmax_adf
from lightning_uq_box.models.adf_layers.adf_utils import convert_deterministic_to_adf

from .base import PosthocBase
from .utils import save_classification_predictions, save_regression_predictions


def keep_variance(x, min_variance):
    return x + min_variance


class ADFClassification(PosthocBase):
    """Assumed Density Filtering.

    If you use this model in your research, please cite the following paper:

    * https://rpg.ifi.uzh.ch/docs/RAL20_Loquercio.pdf
    """

    def __init__(self, model: nn.Module, min_variance: float = 1e-4) -> None:
        """Initialize a new Model instance.

        Args:
            model: pytorch model that will be converted into an adf model
        """

        super().__init__(model)

        self.model = model
        self.min_variance = min_variance
        self.save_hyperparameters(ignore=["model"])
        self._setup_adf_net()

    def _setup_adf_net(self) -> None:
        """Configure setup of the ADF Model."""

        # convert deterministic model to adf net
        convert_deterministic_to_adf(self.model)

    def predict_step(
        self,
        X: Tensor,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        aggregate_fn: Callable = torch.mean,
        eps: float = 1e-7,
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        with torch.no_grad():
            preds = self.model(X)  # shape [batch_size, num_classes, num_classes]

        keep_variance_fn = lambda x: keep_variance(x, min_variance=self.min_variance)

        adf_softmax = Softmax_adf(dim=1, keep_variance_fn=keep_variance_fn)

        logit = [logit for (logit, logit_var) in preds]
        agg_logits = aggregate_fn(logit, dim=-1)
        outputs = [adf_softmax(preds) for outs in preds]
        outputs_mean = [mean for (mean, var) in outputs]
        data_variance = [var for (mean, var) in outputs]
        data_variance = torch.stack(data_variance)
        data_variance = torch.mean(data_variance, dim=0)

        # prevent log of 0 -> nan
        outputs_mean.clamp_min_(eps)
        entropy = -(outputs_mean * outputs_mean.log()).sum(dim=-1)

        return {"pred": outputs_mean, "pred_uct": entropy, "logits": agg_logits}

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

    def adjust_model_logits(self, model_output: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError


class ADFRegression(PosthocBase):
    """Assumed Density Filtering.

    If you use this model in your research, please cite the following paper:

    * https://rpg.ifi.uzh.ch/docs/RAL20_Loquercio.pdf
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize a new Model instance.

        Args:
            model: pytorch model that will be converted into an adf model
        """

        super().__init__(model)

        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._setup_adf_net()

    def _setup_adf_net(self) -> None:
        """Configure setup of the ADF Model."""

        # convert deterministic model to adf net
        convert_deterministic_to_adf(self.model)

    def predict_step(
        self,
        X: Tensor,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        quantiles: Optional[list[float]] = None,
        aggregate_fn: Callable = torch.mean,
    ) -> dict[str, Tensor]:
        """Prediction step and process regression adf predictions.

        Args:
            preds: prediction tensor of shape [batch_size, num_outputs]
            quantiles: quantiles to compute
            aggregate_fn: function to aggregate over the samples to form a mean

        Returns:
            dictionary with mean prediction and predictive uncertainty
        """
        with torch.no_grad():
            preds = self.model(X)  # shape [batch_size, num_outputs]

        mean = aggregate_fn(preds[:, 0], dim=-1)

        # adf yields two num outputs for 1d regression
        if preds.shape[1] == 2:
            aleatoric = preds[:, 1].cpu()  # do we need cpu here

        pred_dict = {"pred": mean, "pred_uct": aleatoric, "aleatoric_uct": aleatoric}

        # check if quantiles are present
        if quantiles is not None:
            quantiles = compute_quantiles_from_std(
                mean.detach().cpu().numpy(), aleatoric, quantiles
            )
            pred_dict["lower_quant"] = torch.from_numpy(quantiles[:, 0])
            pred_dict["upper_quant"] = torch.from_numpy(quantiles[:, -1])

        return pred_dict

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

    def adjust_model_logits(self, model_output: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError
