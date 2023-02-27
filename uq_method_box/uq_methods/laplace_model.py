"""Laplace Approximation model."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class LaplaceModel(LightningModule):
    """Laplace Approximation method for regression."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, **kwargs) -> None:
        """Initialize a new instance of Laplace Model Wrapper."""
        super().__init__()
        self.laplace_fitted = False
        self.map_model = model
        self.train_loader = train_loader
        self.laplace_args = kwargs

    def on_test_start(self) -> None:
        """Fit the Laplace approximation before testing."""
        if not self.laplace_fitted:
            # take the deterministic model we trained and fit laplace
            self.la_model = Laplace(self.map_model, "regression", **self.laplace_args)

            self.la_model.fit(self.train_loader)
            log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True
            )
            # tune the prior precision via Empirical Bayes
            log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True
            )
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
            for i in range(1000):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.la_model.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

            self.laplace_fitted = True

    def test_step(self):
        """Test step with Laplace Approximation."""
        pass

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step with Laplace Approximation."""
        if not self.laplace_fitted:
            self.on_test_start()

        laplace_mean, laplace_var = self.la_model(batch)
        laplace_mean = laplace_mean.squeeze().detach().cpu().numpy()
        laplace_epistemic = laplace_var.squeeze().sqrt().cpu().numpy()
        laplace_aleatoric = self.la_model.sigma_noise.item()
        laplace_predictive = np.sqrt(laplace_epistemic**2 + laplace_aleatoric**2)

        return {
            "mean": laplace_mean,
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
        }
