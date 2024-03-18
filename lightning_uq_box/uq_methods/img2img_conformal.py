# Adapted from https://github.com/aangelopoulos/im2im-uq?tab=readme-ov-file
# MIT License
# Copyright (c) 2021 Anastasios Angelopoulos
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Image-to-Image Conformal Uncertainty Estimation."""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.stats import binom
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .base import PosthocBase
from .utils import (
    default_px_regression_metrics,
    freeze_model_backbone,
    save_image_predictions,
)


class Img2ImgConformal(PosthocBase):
    """Image-to-Image Conformal Uncertainty Estimation.

    This module is a wrapper around a base model that provides
    conformal uncertainty estimates for image-to-image tasks.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2202.05265
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        delta: float = 0.1,
        min_lambda: float = 0.0,
        max_lambda: float = 6.0,
        num_lambdas: int = 100,
        freeze_backbone: bool = False,
    ):
        """Initialize a new Img2ImgConformal instance.

        Args:
            model: the base model to be conformalized
            alpha: the significance level for the conformal prediction
            delta: the
            min_lambda: the minimum lambda value
            max_lambda: the maximum lambda value
            num_lambdas: the number of lambda values to search
            freeze_backbone: whether to freeze the backbone
        """
        super().__init__(model)
        self.delta = delta
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.num_lambdas = num_lambdas

        self.model = model
        self.alpha = alpha
        self.freeze_backbone = freeze_backbone
        self.freeze_model()

        self.setup_task()

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            freeze_model_backbone(self.model)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_px_regression_metrics("train")
        self.val_metrics = default_px_regression_metrics("val")
        self.test_metrics = default_px_regression_metrics("test")

    def forward(self, X: Tensor, lam: Optional[float]) -> dict[str, Tensor]:
        """Forward pass of model.

        Args:
            X: input tensor of shape [batch_size x input_dims]
            lam: The lambda parameter. Default is None.

        Returns:
            model output tensor of shape [batch_size x num_outputs]
        """
        with torch.no_grad():
            if hasattr(self.model, "predict_step"):
                pred = self.model.predict_step(X)
                pred = torch.stack([pred["lower"], pred["pred"], pred["upper"]], dim=1)
            else:
                pred = self.model(X)

        # conformalize in this step
        pred = self.adjust_model_logits(pred, lam)

        return pred

    def adjust_model_logits(self, output: Tensor, lam: Optional[float] = None) -> tuple:
        """Compute the nested sets from the output of the model.

        Args:
            output: The output tensor.
            lam: The lambda parameter. Default is None.

        Returns:
            The lower edge, the output, and the upper edge.
        """
        if lam is None:
            lam = self.lam
        output[:, 0, :, :] = torch.minimum(
            output[:, 0, :, :], output[:, 1, :, :] - 1e-6
        )
        output[:, 2, :, :] = torch.maximum(
            output[:, 2, :, :], output[:, 1, :, :] + 1e-6
        )
        upper_edge = (
            lam * (output[:, 2, :, :] - output[:, 1, :, :]) + output[:, 1, :, :]
        )
        lower_edge = output[:, 1, :, :] - lam * (
            output[:, 1, :, :] - output[:, 0, :, :]
        )

        return {"lower": lower_edge, "pred": output[:, 1, :, :], "upper": upper_edge}

    def rcps_loss_fn(self, pset: dict[str, Tensor], label: Tensor):
        """RCPS Loss function, fraction_missed_loss by default.

        Args:
            pset: The prediction set, output from adjust_model_logits.
            label: The label

        Returns:
            The RCPS loss.
        """
        misses = (pset["lower"].squeeze() > label.squeeze()).float() + (
            pset["upper"].squeeze() < label.squeeze()
        ).float()
        misses[misses > 1.0] = 1.0
        d = len(misses.shape)
        return misses.mean(dim=tuple(range(1, d)))

    def get_rcps_losses_from_outputs(self, logit_dataset: Dataset, lam: float):
        """Compute the RCPS loss from the model outputs.

        Args:
            logit_dataset: The logit dataset.
            rcps_loss_fn: The RCPS loss function.
            lam: The lambda parameter
        """
        losses = []
        dataloader = DataLoader(
            logit_dataset,
            batch_size=self.trainer.datamodule.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for batch in dataloader:
            x, labels = batch
            x = x.to(self.model.device)
            labels = labels.to(self.model.device)
            sets = self.adjust_model_logits(x, lam)
            losses = losses + [self.rcps_loss_fn(sets, labels)]
        return torch.cat(losses, dim=0)

    def on_validation_epoch_end(self) -> None:
        """Perform Img2Img calibration.

        Args:
            outputs: list of dictionaries containing model outputs and labels
        """
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        # probably store all of this in cpu memory instead
        out_dataset = TensorDataset(all_logits, all_labels)

        lambdas = torch.linspace(self.min_lambda, self.max_lambda, self.num_lambdas)

        self.calib_loss_table = torch.zeros((all_labels.shape[0], lambdas.shape[0]))

        dlambda = lambdas[1] - lambdas[0]
        self.lam = lambdas[-1] + dlambda - 1e-9
        for lam in reversed(lambdas):
            losses = self.get_rcps_losses_from_outputs(out_dataset, lam - dlambda)
            self.calib_loss_table[:, np.where(lambdas == lam)[0]] = losses[:, None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], self.delta)
            if Rhat >= self.alpha or RhatPlus > self.alpha:
                self.lam = lam
                break

        self.post_hoc_fitted = True

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()
        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        self.test_metrics(
            pred_dict["pred"].contiguous(), pred_dict[self.target_key].squeeze()
        )

        return pred_dict

    def predict_step(self, X: Tensor, lam: Optional[float] = None) -> dict[str, Tensor]:
        """Prediction step with applied temperature scaling.

        Args:
            X: input tensor of shape [batch_size x num_channels x height x width]
            lam: The lambda parameter
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, "
                "please call trainer.validate(model, datamodule) first."
            )

        return self.forward(X, lam)

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
        save_image_predictions(outputs, batch_idx, self.trainer.default_root_dir)


def h1(y, mu):
    """Compute the h1 function.

    Args:
        y: The value.
        mu: The mean.
    """
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


# Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    """Compute the hoeffding tail inequality for the mean."""
    return -n * h1(np.minimum(mu, x), mu)


def bentkus_plus(mu, x, n):
    """Bentkus tail inequality for the mean.

    Args:
        mu: The mean.
        x: The value.
        n: The number of samples.
    """
    return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1


# UCB of mean via Hoeffding-Bentkus hybridization
def HB_mu_plus(muhat, n, delta, maxiters=1000):
    """Upper Confidence Bound (UCB) of mean via Hoeffding-Bentkus hybridization.

    Args:
        muhat: Estimated mean.
        n: Number of samples.
        delta: Confidence level.
        maxiters: Maximum number of iterations for the brentq method.

    Returns:
        The upper confidence bound of the mean.
    """

    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        try:
            return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)
        except RuntimeError:
            print(f"BRENTQ RUNTIME ERROR at muhat={muhat}")
            return 1.0
