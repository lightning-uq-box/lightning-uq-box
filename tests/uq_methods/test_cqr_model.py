"""Test Conformal Prediction Method."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import CQR, QuantileRegressionModel


class TestCQR:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def qr_model(self, tmp_path: Path) -> QuantileRegressionModel:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "qr_model.yaml"))
        conf.uq_method["save_dir"] = tmp_path

        # train the model with a trainer
        model = instantiate(conf.uq_method)
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

        cqr_model = CQR(
            qr_model, qr_model.hparams.quantiles, save_dir=qr_model.hparams.save_dir
        )
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=cqr_model.hparams.save_dir,
        )
        trainer.test(cqr_model, datamodule)

        return cqr_model

    def test_forward(self, conformalized_model: CQR) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = conformalized_model.num_inputs
        n_outputs = conformalized_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = conformalized_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, conformalized_model: CQR) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = conformalized_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = conformalized_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

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
