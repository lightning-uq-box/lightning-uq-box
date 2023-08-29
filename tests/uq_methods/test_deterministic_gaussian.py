"""Unit tests for Base Model."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import DeterministicGaussianModel


class TestDeterministicGaussianModel:
    @pytest.fixture
    def det_nll_model(self, tmp_path: Path) -> DeterministicGaussianModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "gaussian_nll.yaml"))
        conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_forward(self, det_nll_model: DeterministicGaussianModel) -> None:
        """Test forward pass of base model."""
        n_inputs = det_nll_model.num_inputs
        n_outputs = det_nll_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = det_nll_model(X)
        assert out.shape[-1] == n_outputs

    def test_trainer(self, det_nll_model: DeterministicGaussianModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=2,
            default_root_dir=det_nll_model.hparams.save_dir,
        )
        trainer.fit(model=det_nll_model, datamodule=datamodule)
        trainer.test(model=det_nll_model, datamodule=datamodule)
