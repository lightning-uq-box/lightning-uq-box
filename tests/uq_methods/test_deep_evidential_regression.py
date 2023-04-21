"""Unit tests Deep Evidential Regression."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import DERModel


class TestDERModel:
    @pytest.fixture
    def der_model(self, tmp_path: Path) -> DERModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "der.yaml"))
        conf_dict = OmegaConf.to_object(conf)
        return DERModel(
            MLP, model_args=conf_dict["model"]["model_args"], lr=1e-3, save_dir=tmp_path
        )

    def test_forward(self, der_model: DERModel) -> None:
        """Test forward pass of base model."""
        n_inputs = der_model.hparams.model_args["n_inputs"]
        n_outputs = der_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = der_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, der_model: DERModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = der_model.hparams.model_args["n_inputs"]
        X = torch.randn(5, n_inputs)
        out = der_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, der_model: DERModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=der_model.hparams.save_dir,
        )
        trainer.fit(model=der_model, datamodule=datamodule)
        trainer.test(model=der_model, datamodule=datamodule)
