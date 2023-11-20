"""Test Temperature Scaling."""

import os

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.uq_methods import DeterministicClassification, TempScaling


class TestTempScaling:
    @pytest.fixture
    def deterministic_model(self) -> DeterministicClassification:
        """Create a deterministic model model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "temp_scaling", "temp_scaling.yaml")
        )
        # train the model with a trainer
        model = instantiate(conf.det_method)
        datamodule = TwoMoonsDataModule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture
    def temp_scaled_model(
        self, deterministic_model: DeterministicClassification
    ) -> TempScaling:
        """Apply temperature scaling an underlying model."""
        datamodule = TwoMoonsDataModule()

        temp_scale_model = TempScaling(deterministic_model)
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(temp_scale_model, datamodule)

        return temp_scale_model

    def test_forward(self, temp_scaled_model: TempScaling) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = temp_scaled_model.num_input_features
        n_outputs = temp_scaled_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = temp_scaled_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, temp_scaled_model: TempScaling) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = temp_scaled_model.num_input_features
        X = torch.randn(5, n_inputs)
        out = temp_scaled_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_trainer(self, temp_scaled_model: TempScaling) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = TwoMoonsDataModule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.test(model=temp_scaled_model, datamodule=datamodule)
