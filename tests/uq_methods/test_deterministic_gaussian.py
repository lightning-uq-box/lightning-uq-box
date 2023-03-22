"""Unit tests for Base Model."""

import os
from pathlib import Path

import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import DeterministicGaussianModel


class TestDeterministicGaussianModel:
    @pytest.fixture
    def det_nll_model(self, tmp_path: Path) -> DeterministicGaussianModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "gaussian_nll.yaml"))
        conf_dict = OmegaConf.to_object(conf)
        return DeterministicGaussianModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            loss_fn="nll",
            save_dir=tmp_path,
        )

    def test_forward(self, det_nll_model: DeterministicGaussianModel) -> None:
        """Test forward pass of base model."""
        n_inputs = det_nll_model.hparams.model_args["n_inputs"]
        n_outputs = det_nll_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = det_nll_model(X)
        assert out.shape[-1] == n_outputs

    def test_trainer(self, det_nll_model: DeterministicGaussianModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=det_nll_model.hparams.save_dir,
        )
        trainer.fit(model=det_nll_model, datamodule=datamodule)
        trainer.test(model=det_nll_model, datamodule=datamodule)
