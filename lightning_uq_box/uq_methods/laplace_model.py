# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Laplace Approximation model."""

import os
from typing import Any

import numpy as np
import torch
from laplace import Laplace
from torch import Tensor
from tqdm import trange

from lightning_uq_box.uq_methods import BaseModule

from .utils import (
    _get_num_inputs,
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    save_regression_predictions,
)

# TODO check whether Laplace fitting procedure can be implemented as working
# over training_step in lightning


def tune_prior_precision(
    model: Laplace, tune_precision_lr: float, n_epochs_tune_precision: int
):
    """Tune the prior precision via Empirical Bayes.

    Args:
        model: laplace model
        tune_precision_lr: learning rate for tuning prior precision
        n_epochs_tune_precision: number of epochs to tune prior precision
    """
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
        1, requires_grad=True
    )
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=tune_precision_lr)
    bar = trange(n_epochs_tune_precision)
    # find out why this is so extremely slow?
    for i in bar:
        hyper_optimizer.zero_grad()
        neg_marglik = -model.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
        bar.set_postfix(neg_marglik=f"{neg_marglik.detach().cpu().item()}")


class LaplaceBase(BaseModule):
    """Laplace Approximation Method.

    This is a lightning module wrapper for the `Laplace library <https://aleximmer.github.io/Laplace/>`_. # noqa: E501

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2106.14806
    """

    pred_file_name = "preds.csv"

    def __init__(self, laplace_model: Laplace) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            laplace_model: initialized Laplace model
        """
        super().__init__()

        self.save_hyperparameters(ignore=["laplace_model"])

        self.laplace_model = laplace_model

        self.laplace_fitted = False

        self.setup_task()

    def setup_task(self) -> None:
        """Set up task."""
        pass

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.model.model)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension to model
        """
        return _get_num_outputs(self.model.model)

    # def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    #     pass

    # def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     return super().validation_step(*args, **kwargs)

    # def configure_optimizers(self) -> OptimizerLRScheduler:
    #     pass

    def forward(self, X: Tensor, **kwargs: Any) -> np.ndarray:
        """Fitted Laplace Model Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the laplace model
        """
        if not self.laplace_fitted:
            self.on_test_start()

        return self.laplace_model(X)

    def on_test_start(self) -> None:
        """Fit the Laplace approximation before testing."""
        self.train_loader = self.trainer.datamodule.train_dataloader()

        def collate_fn_laplace_torch(batch):
            """Collate function to for laplace torch tuple convention.

            Args:
                batch: input batch

            Returns:
                renamed batch
            """
            # Extract images and labels from the batch dictionary
            if isinstance(batch[0], dict):
                images = [item[self.input_key] for item in batch]
                labels = [item[self.target_key] for item in batch]
            else:
                images = [item[0] for item in batch]
                labels = [item[1] for item in batch]

            # Stack images and labels into tensors
            inputs = torch.stack(images)
            targets = torch.stack(labels)

            # apply datamodule augmentation
            aug_batch = self.trainer.datamodule.on_after_batch_transfer(
                {self.input_key: inputs, self.target_key: targets}, dataloader_idx=0
            )

            return (aug_batch[self.input_key], aug_batch[self.target_key])

        self.train_loader.collate_fn = collate_fn_laplace_torch

        if not self.laplace_fitted:
            # take the deterministic model we trained and fit laplace
            # laplace needs a nn.Module ant not a lightning module

            # also lightning automatically disables gradient computation during test
            # but need it for laplace so set inference mode to false with cntx manager
            with torch.inference_mode(False):
                # fit the laplace approximation
                self.laplace_model.fit(self.train_loader)

                # tune the prior precision via Empirical Bayes
                self.laplace_model.optimize_prior_precision(method="marglik")
                # tune_prior_precision(
                #     self.model,
                #     self.hparams.tune_precision_lr,
                #     self.hparams.n_epochs_tune_precision,
                # )

            self.laplace_fitted = True

        # save this laplace fitted model as a checkpoint?!

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test step."""
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = (
            batch[self.target_key].detach().squeeze(-1).cpu().numpy()
        )

        self.log(
            "test_loss",
            self.loss_fn(out_dict["pred"], batch[self.target_key].squeeze(-1)),
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key].squeeze(-1))

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in [self.input_key, self.target_key]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        return out_dict

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class LaplaceRegression(LaplaceBase):
    """Laplace Approximation Wrapper for regression.

    This is a lightning module wrapper for the `Laplace library <https://aleximmer.github.io/Laplace/>`_. # noqa: E501

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2106.14806
    """

    def __init__(self, laplace_model: Laplace) -> None:
        """Initialize a new instance of Laplace Model Wrapper for Regression.

        Args:
            laplace_model: initialized Laplace model
        """
        super().__init__(laplace_model)

        assert self.laplace_model.likelihood == "regression"

        self.loss_fn = torch.nn.MSELoss()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Laplace Approximation.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.laplace_fitted:
            self.on_test_start()

        # also lightning automatically disables gradient computation during test
        # but need it for laplace so set inference mode to false with context manager
        with torch.inference_mode(False):
            # inference tensors are not saved for backward so need to create
            # a clone with autograd enables
            input = X.clone().requires_grad_()

            laplace_mean, laplace_var = self.forward(input)
            laplace_mean = laplace_mean.squeeze().detach().cpu().numpy()
            laplace_epistemic = laplace_var.squeeze().sqrt().cpu().numpy()
            laplace_aleatoric = (
                np.ones_like(laplace_epistemic) * self.laplace_model.sigma_noise.item()
            )
            laplace_predictive = np.sqrt(
                laplace_epistemic**2 + laplace_aleatoric**2
            )

        return {
            "pred": torch.from_numpy(laplace_mean).to(self.device),
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
        }

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


class LaplaceClassification(LaplaceBase):
    """Laplace Approximation Wrapper for classification.

    This is a lightning module wrapper for the `Laplace library <https://aleximmer.github.io/Laplace/>`_. # noqa: E501

    If you use this model in your research, please cite the following papers:

    * https://arxiv.org/abs/2106.14806
    """

    valid_tasks = ["binary", "multiclass"]

    def __init__(self, laplace_model: Laplace, task: str = "multiclass") -> None:
        """Initialize a new instance of Laplace Wrapper for Classification.

        Args:
            laplace_model: initialized Laplace model
            task: classification task, one of ['binary', 'multiclass']
        """
        assert task in self.valid_tasks
        self.task = task

        super().__init__(laplace_model)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        assert self.laplace_model.likelihood == "classification"

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, _get_num_outputs(self.laplace_model.model)
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Laplace Approximation.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.laplace_fitted:
            self.on_test_start()

        # also lightning automatically disables gradient computation during test
        # but need it for laplace so set inference mode to false with context manager
        with torch.inference_mode(False):
            # inference tensors are not saved for backward so need to create
            # a clone with autograd enables
            input = X.clone().requires_grad_()

            probs = self.forward(input)

            entropy = -torch.sum(probs * torch.log(probs), dim=1)

        return {"pred": probs, "pred_uct": entropy}
