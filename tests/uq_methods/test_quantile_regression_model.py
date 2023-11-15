"""Test Quantile Regression Model."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import QuantileRegression

# TODO test different quantiles and wrong quantiles


class TestQuantileRegression:
    @pytest.fixture
    def model_regression(self, tmp_path: Path) -> QuantileRegression:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "quantile_regression", "qr_model.yaml")
        )
        return instantiate(conf.uq_method)

    def test_forward(self, model_regression: QuantileRegression) -> None:
        """Test forward pass of base model."""
        n_inputs = model_regression.num_input_dims
        n_outputs = model_regression.num_output_dims
        X = torch.randn(5, n_inputs)
        out = model_regression(X)
        assert out.shape[-1] == n_outputs

    def test_trainer(
        self, model_regression: QuantileRegression, tmp_path: Path
    ) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=2, default_root_dir=str(tmp_path)
        )
        trainer.fit(model=model_regression, datamodule=datamodule)
        trainer.test(model=model_regression, datamodule=datamodule)
