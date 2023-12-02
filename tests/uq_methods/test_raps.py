"""Test RAPS."""

import os

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.uq_methods import RAPS, DeterministicClassification


class TestTempScaling:
    @pytest.fixture
    def deterministic_model(self) -> DeterministicClassification:
        """Create a deterministic model model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "raps", "raps.yaml"))
        # train the model with a trainer
        model = instantiate(conf.det_method)
        datamodule = TwoMoonsDataModule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.validate(model, datamodule.val_dataloader())

        return model

    @pytest.fixture
    def raps_model(self, deterministic_model: DeterministicClassification) -> RAPS:
        """Apply RASP to an underlying model."""
        datamodule = TwoMoonsDataModule()

        temp_scale_model = RAPS(deterministic_model)
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(temp_scale_model, datamodule)

        return temp_scale_model

    def test_forward(self, raps_model: RAPS) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = raps_model.num_input_features
        n_outputs = raps_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = raps_model(X)

    def test_predict_step(self, raps_model: RAPS) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = raps_model.num_input_features
        X = torch.randn(5, n_inputs)
        out = raps_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5
