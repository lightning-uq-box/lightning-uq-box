"""Laplace Approximation model."""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.trainers.utils import _get_input_layer_name_and_module
from tqdm import trange

from uq_method_box.eval_utils import compute_quantiles_from_std

from .utils import _get_output_layer_name_and_module, save_predictions_to_csv


class LaplaceModel(LightningModule):
    """Laplace Approximation method for regression."""

    subset_of_weights_options = ["last_layer", "subnetwork", "all"]
    hessian_structure_options = ["diag", "kron", "full", "lowrank"]

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        save_dir: str,
        sigma_noise: float = 1.0,
        prior_precision: float = 1.0,
        prior_mean: float = 0.0,
        temperature: float = 1.0,
        subset_of_weights: str = "last_layer",
        hessian_structure: str = "kron",
        tune_precision_lr: float = 0.1,
        n_epochs_tune_precision: int = 100,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            model: pytorch model to use as underlying model
            laplace_args: laplace arguments to initialize a Laplace Model
            train_loader: train loader to be used but maybe this can
                also be accessed through the trainer or write a
                train_dataloader() method for this model based on the config?
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_loader"])

        self.model = model  # pytorch model
        self.train_loader = train_loader
        self.laplace_fitted = False

    @property
    def num_inputs(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_input_layer_name_and_module(self.model)
        if hasattr(module, "in_features"):  # Linear Layer
            num_inputs = module.in_features
        elif hasattr(module, "in_channels"):  # Conv Layer
            num_inputs = module.in_channels
        return num_inputs

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_output_layer_name_and_module(self.model)
        if hasattr(module, "out_features"):  # Linear Layer
            num_outputs = module.out_features
        elif hasattr(module, "out_channels"):  # Conv Layer
            num_outputs = module.out_channels
        return num_outputs

    def forward(self, X: Tensor, **kwargs: Any) -> np.ndarray:
        """Fitted Laplace Model Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the laplace model
        """
        if not self.laplace_fitted:
            self.on_test_start()

        return self.la_model(X)

    def on_test_start(self) -> None:
        """Fit the Laplace approximation before testing."""
        if not self.laplace_fitted:
            # take the deterministic model we trained and fit laplace
            # laplace needs a nn.Module ant not a lightning module

            # also lightning automatically disables gradient computation during test
            # but need it for laplace so set inference mode to false with cntx manager
            with torch.inference_mode(False):
                self.la_model = Laplace(
                    model=self.model,
                    likelihood="regression",
                    subset_of_weights=self.hparams.subset_of_weights,
                    hessian_structure=self.hparams.hessian_structure,
                    sigma_noise=self.hparams.sigma_noise,
                    prior_precision=self.hparams.prior_precision,
                    prior_mean=self.hparams.prior_mean,
                    temperature=self.hparams.temperature,
                )
                # fit the laplace approximation
                self.la_model.fit(self.train_loader)

                # tune the prior precision via Empirical Bayes
                log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
                    1, requires_grad=True
                )
                hyper_optimizer = torch.optim.Adam(
                    [log_prior, log_sigma], lr=self.hparams.tune_precision_lr
                )
                bar = trange(self.hparams.n_epochs_tune_precision)
                # find out why this is so extremely slow?
                for i in bar:
                    hyper_optimizer.zero_grad()
                    neg_marglik = -self.la_model.log_marginal_likelihood(
                        log_prior.exp(), log_sigma.exp()
                    )
                    neg_marglik.backward()
                    hyper_optimizer.step()
                    bar.set_postfix(neg_marglik=f"{neg_marglik.detach().cpu().item()}")

            self.laplace_fitted = True

        # save this laplace fitted model as a checkpoint?!

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.detach().squeeze(-1).numpy()
        return out_dict

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
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
    ) -> dict[str, np.ndarray]:
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
                np.ones_like(laplace_epistemic) * self.la_model.sigma_noise.item()
            )
            laplace_predictive = np.sqrt(
                laplace_epistemic**2 + laplace_aleatoric**2
            )
            quantiles = compute_quantiles_from_std(
                laplace_mean, laplace_predictive, self.hparams.quantiles
            )

        return {
            "mean": laplace_mean,
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
