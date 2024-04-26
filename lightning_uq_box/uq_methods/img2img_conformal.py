# Adapted from https://github.com/aangelopoulos/im2im-uq?tab=readme-ov-file
# to be compatible with Lightning.
# MIT License
# Copyright (c) 2021 Anastasios Angelopoulos

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Image-to-Image Conformal Uncertainty Estimation."""

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from scipy.optimize import brentq
from scipy.stats import binom, spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .base import PosthocBase
from .utils import default_px_regression_metrics, save_image_predictions


class Img2ImgConformal(PosthocBase):
    """Image-to-Image Conformal Uncertainty Estimation.

    This module is a wrapper around a base model that provides
    conformal uncertainty estimates for image-to-image tasks, as
    introduced by `Angelopoulos et al. (2022) <https://arxiv.org/abs/2202.05265>`_.

    This default implementation uses the quantile regression
    approach, as it demonstrated good results in the original paper.

    But other approaches can be implemented as well, by using a
    different architecture and overwriting `adjust_model_logits` and
    `rcps_loss_fn` methods.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2202.05265
    """

    pred_dir_name = "preds"
    metrics_file_name = "test_metrics.json"

    def __init__(
        self,
        model: LightningModule | nn.Module,
        alpha: float = 0.1,
        delta: float = 0.1,
        min_lambda: float = 0.0,
        max_lambda: float = 6.0,
        num_lambdas: int = 1000,
    ):
        """Initialize a new Img2ImgConformal instance.

        The user can select a risk level α ∈ (0, 1), and an error level
        δ ∈ (0, 1), such as α = δ = 0.1. Then, the conformal procedure construct
        intervals that contain at least 1 − α of the ground-truth pixel values
        with probability 1 − δ.

        Args:
            model: the base model to be conformalized
            alpha: the risk level
            delta: the error level
            min_lambda: the minimum lambda value
            max_lambda: the maximum lambda value
            num_lambdas: the number of lambda values to search
        """
        super().__init__(model)
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.num_lambdas = num_lambdas

        self.model = model
        assert alpha > 0 and alpha < 1, "alpha must be in (0, 1)"
        assert delta > 0 and delta < 1, "delta must be in (0, 1)"
        self.delta = delta
        self.alpha = alpha

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_px_regression_metrics("test")

    def forward(self, X: Tensor, lam: float | None) -> dict[str, Tensor]:
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
                pred = self.model(X).squeeze(2)

        # conformalize in this step
        pred = self.adjust_model_logits(pred, lam)

        # authors define set size as the uncertainty
        pred["pred_uct"] = pred["upper"] - pred["lower"]

        return pred

    def adjust_model_logits(self, output: Tensor, lam: float | None = None) -> tuple:
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

    @torch.no_grad()
    def get_rcps_losses_from_outputs(self, logit_dataset: Dataset, lam: float):
        """Compute the RCPS loss from the model outputs.

        Args:
            logit_dataset: The logit dataset.
            rcps_loss_fn: The RCPS loss function.
            lam: The lambda parameter
        """
        losses = []
        dataloader = DataLoader(
            logit_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            output, labels = batch
            sets = self.adjust_model_logits(output, lam)
            losses = losses + [self.rcps_loss_fn(sets, labels)]

        return torch.cat(losses, dim=0).cpu()

    @torch.no_grad()
    def on_train_end(self) -> None:
        """Perform Img2Img calibration."""
        self.eval()
        self.batch_size = self.model_logits[0].shape[0]
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        # probably store all of this in cpu memory instead
        self.out_dataset = TensorDataset(all_logits, all_labels)

        lambdas = torch.linspace(self.min_lambda, self.max_lambda, self.num_lambdas)

        self.calib_loss_table = torch.zeros((all_labels.shape[0], lambdas.shape[0]))

        dlambda = lambdas[1] - lambdas[0]
        self.lam = lambdas[-1] + dlambda - 1e-9
        for lam in reversed(lambdas):
            losses = self.get_rcps_losses_from_outputs(self.out_dataset, lam - dlambda)
            self.calib_loss_table[:, np.where(lambdas == lam)[0]] = losses[:, None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], self.delta)
            if Rhat >= self.alpha or RhatPlus > self.alpha:
                self.lam = lam
                break

        # save the calibration table to log_dir
        np.save(
            os.path.join(self.trainer.default_root_dir, "calib_loss_table.npy"),
            self.calib_loss_table.numpy(),
        )

        self.post_hoc_fitted = True

    def on_test_start(self) -> None:
        """Create logging directory and initialize metrics."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

        # Initialize metrics
        self.losses = []
        self.sizes = []
        self.residuals = []
        self.spatial_miscoverages = []

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            test step dictionary with predictions
        """
        pred_dict = self.predict_step(batch[self.input_key])

        # torchmetrics
        self.test_metrics(
            pred_dict["pred"].contiguous().squeeze(), batch[self.target_key].squeeze()
        )

        # Compute our metrics
        self.compute_metrics(pred_dict, batch[self.target_key].squeeze(1))

        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()
        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        return pred_dict

    @torch.no_grad()
    def compute_metrics(self, pred_dict: dict[str, Tensor], labels: Tensor) -> None:
        """Compute metrics for a batch and append them to the metrics lists.

        Args:
            pred_dict: The prediction dictionary from predict_step for a batch
            labels: The corresponding batch labels
        """
        losses = self.rcps_loss_fn(pred_dict, labels)
        self.losses.append(losses.cpu())

        sets_full = (pred_dict["upper"] - pred_dict["lower"]).flatten(start_dim=1)
        # need to take random idxs, because later on during aggregation
        # quantile is a limiting factor
        rng = np.random.RandomState(0)
        size_random_idxs = rng.choice(sets_full.shape[1], size=sets_full.shape[0])
        size_samples = sets_full[range(sets_full.shape[0]), size_random_idxs]
        self.sizes.append(size_samples.cpu())

        residuals = (
            (labels - pred_dict["pred"])
            .abs()
            .flatten(start_dim=1)[range(sets_full.shape[0]), size_random_idxs]
        )
        self.residuals.append(residuals.cpu())

        spatial_miscoverages = (labels > pred_dict["upper"]).float() + (
            labels < pred_dict["lower"]
        ).float()
        self.spatial_miscoverages.append(spatial_miscoverages.cpu())

    def on_test_end(self) -> None:
        """Summarize metrics."""
        self.losses = torch.cat(self.losses, dim=0)
        sizes = torch.cat(self.sizes, dim=0)
        self.sizes = sizes + torch.rand_like(sizes) * 1e-6
        self.residuals = torch.cat(self.residuals, dim=0)

        spatial_miscoverages = torch.cat(self.spatial_miscoverages, dim=0)

        self.spatial_miscoverages = {
            "mean": spatial_miscoverages.mean(dim=[1, 2]).mean(),
            "std": spatial_miscoverages.std(dim=[1, 2]).mean(),
            "min": spatial_miscoverages.min(),
            "max": spatial_miscoverages.max(),
        }
        size_bins = torch.tensor(
            [
                0,
                torch.quantile(self.sizes, 0.25),
                torch.quantile(self.sizes, 0.5),
                torch.quantile(self.sizes, 0.75),
            ]
        )
        buckets = torch.bucketize(self.sizes, size_bins) - 1
        stratified_risks = torch.tensor(
            [
                self.losses[buckets == bucket].mean()
                for bucket in range(size_bins.shape[0])
            ]
        )

        # now can aggregate the metrics
        spearman = spearmanr(self.residuals, self.sizes)[0]
        mse = (self.residuals * self.residuals).mean()
        losses = self.losses.mean()
        sizes = self.sizes.mean()
        res = self.residuals.mean()

        # save all these metrics to a file
        metrics = {
            "risk": losses.item(),
            "Sizes": sizes.item(),
            "Spearman": spearman,
            "Stratified Risks": stratified_risks.tolist(),
            "Size bins": size_bins.tolist(),
            "MSE": mse.item(),
            "Residuals": res.item(),
            "Spatial Miscoverage": {
                key: val.item() for key, val in self.spatial_miscoverages.items()
            },
        }

        with open(
            os.path.join(self.trainer.default_root_dir, self.metrics_file_name), "w"
        ) as f:
            json.dump(metrics, f)

    def predict_step(self, X: Tensor, lam: float | None = None) -> dict[str, Tensor]:
        """Prediction step with applied temperature scaling.

        Args:
            X: input tensor of shape [batch_size x num_channels x height x width]
            lam: The lambda parameter

        Returns:
            prediction dictionary
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, "
                "please call "
                "trainer.fit(model, train_dataloaders=dm.calib_dataloader()) first."
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
        save_image_predictions(outputs, batch_idx, self.pred_dir)


def h1(y: "np.typing.NDArray[np.float32]", mu: "np.typing.NDArray[np.float32]"):
    """Compute the h1 function.

    Args:
        y: The value.
        mu: The mean.
    """
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


# Log tail inequalities of mean
def hoeffding_plus(
    mu: "np.typing.NDArray[np.float32]", x: "np.typing.NDArray[np.float32]", n: int
):
    """Compute the hoeffding tail inequality for the mean."""
    return -n * h1(np.minimum(mu, x), mu)


def bentkus_plus(
    mu: "np.typing.NDArray[np.float32]", x: "np.typing.NDArray[np.float32]", n: int
):
    """Bentkus tail inequality for the mean.

    Args:
        mu: The mean.
        x: The value.
        n: The number of samples.
    """
    return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1


# UCB of mean via Hoeffding-Bentkus hybridization
def HB_mu_plus(
    muhat: "np.typing.NDArray[np.float32]", n: int, delta: float, maxiters: int = 1000
):
    """Upper Confidence Bound (UCB) of mean via Hoeffding-Bentkus hybridization.

    Args:
        muhat: Estimated mean.
        n: Number of samples.
        delta: the error level
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
