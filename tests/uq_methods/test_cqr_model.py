"""Test Conformal Prediction Method."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import ConformalQR, QuantileRegression

# TODO need to test that we are able to conformalize other models


class TestCQR:
    @pytest.fixture
    def qr_model(self, tmp_path: Path) -> QuantileRegression:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "quantile_regression", "qr_model.yaml")
        )

        # train the model with a trainer
        model = instantiate(conf.uq_method)
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture
    def conformalized_model(self, qr_model: QuantileRegression) -> ConformalQR:
        """Conformalize an underlying model."""
        datamodule = ToyHeteroscedasticDatamodule()

        cqr_model = ConformalQR(qr_model, qr_model.hparams.quantiles)
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.validate(cqr_model, datamodule.val_dataloader())

        return cqr_model

    def test_forward(self, conformalized_model: ConformalQR) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = conformalized_model.num_input_features
        n_outputs = len(conformalized_model.quantiles)
        X = torch.randn(5, n_inputs)
        out = conformalized_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, conformalized_model: ConformalQR) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = conformalized_model.num_input_features
        X = torch.randn(5, n_inputs)
        out = conformalized_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_trainer(self, conformalized_model: ConformalQR) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.test(model=conformalized_model, datamodule=datamodule)
