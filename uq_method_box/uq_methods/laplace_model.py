"""Laplace Approximation model."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from uq_method_box.eval_utils import compute_quantiles_from_std
from uq_method_box.uq_methods import BaseModel


class LaplaceModel(BaseModel):
    """Laplace Approximation method for regression."""

    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper."""
        super().__init__(config, model, criterion)
        self.laplace_fitted = False
        self.train_loader = train_loader

        # get laplace args from dictionary
        self.laplace_args = {
            arg: val
            for arg, val in self.config["model"]["laplace"].items()
            if arg not in ["n_epochs_tune_precision", "tune_precision_lr"]
        }
        self.tune_precision_lr = self.config["model"]["laplace"].get(
            "tune_precision_lr", 1e-2
        )
        self.n_epochs_tune_precision = self.config["model"]["laplace"].get(
            "n_epochs_tune_precision", 100
        )

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert (
            out.shape[-1] == 1
        ), "This model should give exactly 1 outputs due to MAP estimation."
        return out

    def on_test_start(self) -> None:
        """Fit the Laplace approximation before testing."""
        if not self.laplace_fitted:
            # take the deterministic model we trained and fit laplace
            # laplace needs a nn.Module ant not a lightning module

            # also lightning automatically disables gradient computation during test
            # but need it for laplace so set inference mode to false with cntx manager
            with torch.inference_mode(False):
                self.la_model = Laplace(
                    self.model.model, "regression", **self.laplace_args
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

    def test_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Test step with Laplace Approximation.

        Args:
            batch:

        Returns:
            dictionary of uncertainty outputs
        """
        target = batch[1]
        out_dict = self.predict_step(batch)
        out_dict["targets"] = target.detach().squeeze(-1).numpy()
        return out_dict

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step with Laplace Approximation."""
        if not self.laplace_fitted:
            self.on_test_start()

        # also lightning automatically disables gradient computation during test
        # but need it for laplace so set inference mode to false with context manager
        with torch.inference_mode(False):
            # inference tensors are not saved for backward so need to create
            # a clone with autograd enables
            input = batch[0].clone().requires_grad_()

            laplace_mean, laplace_var = self.la_model(input)
            laplace_mean = laplace_mean.squeeze().detach().cpu().numpy()
            laplace_epistemic = laplace_var.squeeze().sqrt().cpu().numpy()
            laplace_aleatoric = (
                np.ones_like(laplace_epistemic) * self.la_model.sigma_noise.item()
            )
            laplace_predictive = np.sqrt(
                laplace_epistemic**2 + laplace_aleatoric**2
            )
            quantiles = compute_quantiles_from_std(
                laplace_mean,
                laplace_predictive,
                self.config["model"].get("quantiles", [0.1, 0.5, 0.9]),
            )

        return {
            "mean": laplace_mean,
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
