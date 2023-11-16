"""Unit tests Deep Evidential Regression."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import DER


class TestDERModel:
    @pytest.fixture
    def der_model(self, tmp_path: Path) -> DER:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "deep_evidential_regression", "der.yaml")
        )
        return instantiate(conf.uq_method)

    def test_forward(self, der_model: DER) -> None:
        """Test forward pass of base model."""
        n_inputs = der_model.num_input_dims
        n_outputs = der_model.num_output_dims
        X = torch.randn(5, n_inputs)
        out = der_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, der_model: DER) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = der_model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = der_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_trainer(self, der_model: DER) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=der_model, datamodule=datamodule)
        trainer.test(model=der_model, datamodule=datamodule)
