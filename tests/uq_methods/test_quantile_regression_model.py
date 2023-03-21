"""Test Quantile Regression Model."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import QuantileRegressionModel

# TODO test different quantiles and wrong quantiles


class TestQuantileRegressionModel:
    @pytest.fixture
    def qr_model(self, tmp_path: Path) -> QuantileRegressionModel:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "qr_model.yaml"))
        conf_dict = OmegaConf.to_object(conf)
        return QuantileRegressionModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            save_dir=tmp_path,
            quantiles=conf_dict["model"]["quantiles"],
        )

    def test_forward(self, qr_model: QuantileRegressionModel) -> None:
        """Test forward pass of QR model."""
        n_inputs = qr_model.hparams.model_args["n_inputs"]
        n_outputs = qr_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = qr_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, qr_model: QuantileRegressionModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = qr_model.hparams.model_args["n_inputs"]
        X = torch.randn(5, n_inputs)
        out = qr_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, qr_model: QuantileRegressionModel) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=qr_model.hparams.save_dir,
        )
        trainer.fit(model=qr_model, datamodule=datamodule)
        trainer.test(model=qr_model, datamodule=datamodule)
