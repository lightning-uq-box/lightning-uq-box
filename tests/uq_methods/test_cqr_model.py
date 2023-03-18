"""Test Conformal Prediction Method."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer

# required to make the path visible to import the tools
# this will change in public notebooks to be "pip install uq-regression-box"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import CQR, QuantileRegressionModel


class TestCQR:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def qr_model(self, tmp_path: Path) -> QuantileRegressionModel:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "qr_model.yaml"))
        conf_dict = OmegaConf.to_object(conf)

        # train the model with a trainer
        model = QuantileRegressionModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            save_dir=tmp_path,
            quantiles=conf_dict["model"]["quantiles"],
        )
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=model.hparams.save_dir
        )
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture
    def conformalized_model(self, qr_model) -> CQR:
        """Conformalize an underlying model."""
        datamodule = ToyHeteroscedasticDatamodule()
        calib_loader = datamodule.val_dataloader()
        return CQR(
            qr_model,
            qr_model.quantiles,
            calib_loader,
            save_dir=qr_model.hparams.save_dir,
        )

    def test_forward(self, conformalized_model: CQR) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = conformalized_model.model.hparams.model_args["n_inputs"]
        n_outputs = conformalized_model.model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = conformalized_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, conformalized_model: CQR) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = conformalized_model.model.hparams.model_args["n_inputs"]
        X = torch.randn(5, n_inputs)
        out = conformalized_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, conformalized_model: CQR) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=conformalized_model.hparams.save_dir,
        )
        trainer.test(model=conformalized_model, datamodule=datamodule)
