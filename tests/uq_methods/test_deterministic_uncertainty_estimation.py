"""Deterministic Uncertainty Estimation."""

import os
from pathlib import Path

import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest_lazyfixture import lazy_fixture
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import DUERegression


class TestDUEModel:
    @pytest.fixture
    def model_regression(self, tmp_path: Path) -> DUERegression:
        """Create DUE model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join(
                "tests",
                "configs",
                "deterministic_uncertainty_estimation",
                "due_regression.yaml",
            )
        )

        due_model = instantiate(conf.uq_method)
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(due_model, datamodule=ToyHeteroscedasticDatamodule())

        return due_model

    @pytest.mark.parametrize("model", [lazy_fixture("model_regression")])
    def test_forward(self, model: DUERegression) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = model(X)
        assert isinstance(out, MultivariateNormal)

    @pytest.mark.parametrize("model", [lazy_fixture("model_regression")])
    def test_predict_step(self, model: DUERegression) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        # backpack expects a torch.nn.sequential but also works otherwise
        out = model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5
