"""Test Deep Kernel Learning Model."""

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
from lightning_uq_box.uq_methods import DKLRegression


class TestDeepKernelLearningModel:
    @pytest.fixture
    def model_regression(self, tmp_path: Path) -> DKLRegression:
        """Create DKL model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join(
                "tests", "configs", "deep_kernel_learning", "dkl_regression.yaml"
            )
        )
        dkl_model = instantiate(conf.uq_method)
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(dkl_model, datamodule=ToyHeteroscedasticDatamodule())

        return dkl_model

    @pytest.mark.parametrize("model", [lazy_fixture("model_regression")])
    def test_forward(self, model: DKLRegression) -> None:
        """Test forward pass."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = model(X)
        assert isinstance(out, MultivariateNormal)

    @pytest.mark.parametrize("model", [lazy_fixture("model_regression")])
    def test_predict_step(self, model: DKLRegression) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5
