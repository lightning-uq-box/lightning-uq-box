"""Laplace Approximation model."""

import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from laplace import Laplace
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from uq_method_box.eval_utils import compute_quantiles_from_std

from .utils import save_predictions_to_csv


class LaplaceModel(LightningModule):
    """Laplace Approximation method for regression."""

    def __init__(
        self,
        model: LightningModule,
        laplace_args: Dict[str, Union[float, str]],
        train_loader: DataLoader,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            model: lightning module to use as underlying model
            laplace_args: laplace arguments to initialize a Laplace Model
            train_loader: train loader to be used but maybe this can
                also be accessed through the trainer or write a
                train_dataloader() method for this model based on the config?
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_loader"])
        self.laplace_fitted = False
        self.train_loader = train_loader

        self.model = model  # lightning model

        # TODO: find the sequential within the model
        self.module = self.model.model.model  # nn.Sequential

        self.tune_precision_lr = self.hparams.laplace_args.get(
            "tune_precision_lr", 1e-2
        )
        self.n_epochs_tune_precision = self.hparams.laplace_args.get(
            "n_epochs_tune_precision", 100
        )
        # get laplace args from dictionary
        self.hparams.laplace_args = {
            arg: val
            for arg, val in self.hparams.laplace_args.items()
            if arg not in ["n_epochs_tune_precision", "tune_precision_lr"]
        }

        self.quantiles = quantiles

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
                    self.module, "regression", **self.hparams.laplace_args
                )
                # fit the laplace approximation
                self.la_model.fit(self.train_loader)

                # tune the prior precision via Empirical Bayes
                log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
                    1, requires_grad=True
                )
                hyper_optimizer = torch.optim.Adam(
                    [log_prior, log_sigma], lr=self.tune_precision_lr
                )
                bar = trange(self.n_epochs_tune_precision)
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
        X, y_true = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y_true.detach().squeeze(-1).numpy()
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
                laplace_mean, laplace_predictive, self.quantiles
            )

        return {
            "mean": laplace_mean,
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
