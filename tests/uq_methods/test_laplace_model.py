"""Test Laplace Model."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import BaseModel, LaplaceModel

# TODO need to test all different laplace args


class TestLaplaceModel:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def base_model(self, tmp_path: Path) -> BaseModel:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "laplace.yaml"))
        conf.uq_method["save_dir"] = str(tmp_path)

        # train the model with a trainer
        model = instantiate(conf.uq_method)
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=model.save_dir
        )
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture
    def laplace_model(self, base_model: BaseModel, tmp_path: Path) -> LaplaceModel:
        """Create Laplace model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "laplace.yaml"))
        conf.post_processing["save_dir"] = str(tmp_path)
        laplace_model = instantiate(conf.post_processing, model=base_model.model)
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=laplace_model.hparams.save_dir,
        )
        trainer.test(laplace_model, datamodule=ToyHeteroscedasticDatamodule())
        return laplace_model

    def test_forward(self, laplace_model: LaplaceModel) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = laplace_model.num_inputs
        X = torch.randn(5, n_inputs)
        # output of laplace like it is in the libray
        out = laplace_model(X)
        assert isinstance(out, tuple)
        assert out[0].shape[-1] == 1
        assert out[-1].shape[-1] == 1

    def test_predict_step(self, laplace_model: LaplaceModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = laplace_model.num_inputs
        X = torch.randn(5, n_inputs)
        # backpack expects a torch.nn.sequential but also works otherwise
        out = laplace_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], np.ndarray)
        assert out["pred"].shape[0] == 5

    def test_trainer(self, laplace_model: LaplaceModel) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=laplace_model.hparams.save_dir,
        )
        # backpack expects a torch.nn.sequential but also works otherwise
        trainer.test(model=laplace_model, datamodule=datamodule)
