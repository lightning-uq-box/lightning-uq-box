"""Test Deterministic Uncertainty Model."""

import os
from pathlib import Path
from typing import Union

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest_lazyfixture import lazy_fixture
from torch import Tensor

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.uq_methods import DUEClassification, DUERegression


class TestDUE:
    @pytest.fixture
    def model_regression(self, tmp_path: Path) -> DUERegression:
        """Create DUE model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join(
                "tests",
                "configs",
                "deterministic_uncertainty_estimation",
                "due_regression.yaml",
            )
        )
        dkl_model = instantiate(conf.uq_method)
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(dkl_model, datamodule=ToyHeteroscedasticDatamodule())

        return dkl_model

    @pytest.fixture
    def model_classification(self, tmp_path: Path) -> DUERegression:
        """Create DUE model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join(
                "tests",
                "configs",
                "deterministic_uncertainty_estimation",
                "due_classification.yaml",
            )
        )
        dkl_model = instantiate(conf.uq_method)
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(dkl_model, datamodule=TwoMoonsDataModule())

        return dkl_model

    @pytest.mark.parametrize(
        "model",
        [lazy_fixture("model_regression"), lazy_fixture("model_classification")],
    )
    def test_forward(self, model: Union[DUERegression, DUEClassification]) -> None:
        """Test forward pass of DUE model."""
        n_inputs = model.num_input_features
        n_outputs = model.num_outputs
        X = torch.randn(5, n_inputs)
        out = model(X)

    @pytest.mark.parametrize(
        "model",
        [lazy_fixture("model_regression"), lazy_fixture("model_classification")],
    )
    def test_predict_step(self, model: Union[DUERegression, DUEClassification]) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = model.num_input_features
        X = torch.randn(5, n_inputs)
        out = model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5
