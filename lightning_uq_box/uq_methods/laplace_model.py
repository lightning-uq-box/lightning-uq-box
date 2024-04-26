# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Laplace Approximation model."""

import copy
import os
from typing import Any

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
    save_classification_predictions,
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
    log_prior, log_sigma = (
        torch.ones(1, requires_grad=True),
        torch.ones(1, requires_grad=True),
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

    def __init__(
        self,
        laplace_model: Laplace,
        pred_type: str = "glm",
        link_approx: str = "probit",
        num_samples: int | None = None,
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            laplace_model: initialized Laplace model
            pred_type: prediction type, one of ['glm', 'nn']
            link_approx: link function approximation, one of ['mc', 'probit', 'bridge']
                for `pred_type='nn'` only 'mc' is supported
            num_samples: number of samples for prediction, if specified
                will call `predictive_samples` instead of `predictive` method in
                Laplace library
        """
        super().__init__()

        if pred_type == "nn":
            assert link_approx == "mc", "For nn prediction only mc link is supported"

        self.pred_type = pred_type
        self.link_approx = link_approx
        self.num_samples = num_samples

        self.save_hyperparameters(ignore=["laplace_model"])

        # reinitialize the model with the correct device because cannot set device
        # to laplace model afterwards
        LaplaceClass = type(laplace_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_args = laplace_model.__init__.__code__.co_varnames
        model_copy = copy.deepcopy(laplace_model.model)
        model = model_copy.to(device)

        args_dict = {
            arg: getattr(laplace_model, arg)
            for arg in init_args
            if hasattr(laplace_model, arg) and arg != "model"
        }
        args_dict["model"] = model

        # Create a new instance of the LaplaceClass with the same arguments,
        # but with the model on the CUDA device
        self.laplace_model = LaplaceClass(**args_dict)

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
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()

        self.log(
            "test_loss",
            self.loss_fn(out_dict["pred"], batch[self.target_key].squeeze(-1)),
            batch_size=batch[self.input_key].shape[0],
        )  # logging to Logger
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch[self.target_key].squeeze(-1))

        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        # save metadata
        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        return out_dict

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class LaplaceRegression(LaplaceBase):
    """Laplace Approximation Wrapper for regression.

    This is a lightning module wrapper for the
    `Laplace library <https://aleximmer.github.io/Laplace/>`_.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2106.14806
    """

    def __init__(
        self,
        laplace_model: Laplace,
        pred_type: str = "glm",
        link_approx: str = "probit",
        num_samples: int | None = None,
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper for regression.

        Args:
            laplace_model: initialized Laplace model
            pred_type: prediction type, one of ['glm', 'nn']
            link_approx: link function approximation, one of ['mc', 'probit', 'bridge']
                for `pred_type='nn'` only 'mc' is supported
            num_samples: number of samples for prediction, if specified
                will call `predictive_samples` instead of `predictive` method in
                Laplace library
        """
        super().__init__(laplace_model, pred_type, link_approx, num_samples)

        assert self.laplace_model.likelihood == "regression"

        self.loss_fn = torch.nn.MSELoss()

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_regression_metrics("test")

    def forward(self, X: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Fitted Laplace Model Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]
            kwargs: additional arguments for laplace forward pass

        Returns:
            output from the laplace model
        """
        if not self.laplace_fitted:
            self.on_test_start()

        pred_dict: dict[str, Tensor] = {}
        if self.num_samples:
            fsamples = self.laplace_model.predictive_samples(
                X, pred_type=self.pred_type, n_samples=self.num_samples
            )
            mean = fsamples.mean(0).squeeze()
            pred_std = fsamples.std(0).squeeze()
            # return samples as shape [batch_size, out_dim, num_samples]
            pred_dict["samples"] = fsamples.permute(1, 2, 0)
        else:
            mean, var = self.laplace_model(
                X, pred_type=self.pred_type, link_approx=self.link_approx
            )
            mean = mean.squeeze().detach()
            laplace_epistemic = var.squeeze().sqrt()

            laplace_aleatoric = (
                torch.ones_like(laplace_epistemic)
                * self.laplace_model.sigma_noise.item()
            )
            pred_std = torch.sqrt(laplace_epistemic**2 + laplace_aleatoric**2)

            pred_dict["epistemic"] = laplace_epistemic
            pred_dict["aleatoric"] = laplace_aleatoric

        pred_dict["pred"] = mean
        pred_dict["pred_uct"] = pred_std
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with Laplace Approximation.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

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

        return self.forward(input)

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

    This is a lightning module wrapper for the
    `Laplace library <https://aleximmer.github.io/Laplace/>`_.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2106.14806
    """

    valid_tasks = ["binary", "multiclass"]

    def __init__(
        self,
        laplace_model: Laplace,
        task: str = "multiclass",
        pred_type: str = "glm",
        link_approx: str = "probit",
        num_samples: int | None = None,
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper for Classification.

        Args:
            laplace_model: initialized Laplace model
            task: classification task, one of ['binary', 'multiclass']
            pred_type: prediction type, one of ['glm', 'nn']
            link_approx: link function approximation, one of ['mc', 'probit', 'bridge']
                for `pred_type='nn'` only 'mc' is supported
            num_samples: number of samples for prediction, if specified
                will call `predictive_samples` instead of `predictive` method in
                Laplace library
        """
        assert task in self.valid_tasks
        self.task = task

        super().__init__(laplace_model, pred_type, link_approx, num_samples)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        assert self.laplace_model.likelihood == "classification"

    def forward(self, X: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Fitted Laplace Model Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]
            kwargs: additional arguments for laplace forward pass

        Returns:
            output from the laplace model
        """
        if not self.laplace_fitted:
            self.on_test_start()

        pred_dict: dict[str, Tensor] = {}
        if self.num_samples:
            fsamples = self.laplace_model.predictive_samples(
                X, pred_type=self.pred_type, n_samples=self.num_samples
            )
            mean = fsamples.mean(0)
            pred_dict["samples"] = fsamples
        else:
            mean = self.laplace_model(
                X, pred_type=self.pred_type, link_approx=self.link_approx
            )
        pred_dict["pred"] = mean
        return pred_dict

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
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

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

        return {"pred": probs, "pred_uct": entropy, "logits": probs}

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
