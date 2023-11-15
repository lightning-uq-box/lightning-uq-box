"""Test Laplace Model."""

import os
from pathlib import Path
from typing import Union

import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest_lazyfixture import lazy_fixture
from torch import Tensor

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.uq_methods import (
    DeterministicClassification,
    DeterministicModel,
    LaplaceClassification,
    LaplaceRegression,
)


class TestLaplace:
    @pytest.fixture(params=["laplace_classification.yaml", "laplace_regression.yaml"])
    def model(
        self, request: SubRequest
    ) -> Union[LaplaceRegression, LaplaceClassification]:
        """Create a Laplace Model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "laplace", request.param)
        )
        deterministic_model = instantiate(conf.model)

        laplace_model = instantiate(conf.laplace, model=deterministic_model)

        laplace_module = instantiate(conf.uq_method, model=laplace_model)
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)

        if isinstance(laplace_module, LaplaceRegression):
            datamodule = ToyHeteroscedasticDatamodule()
        else:
            datamodule = TwoMoonsDataModule()

        trainer.test(laplace_module, datamodule=datamodule)

        return laplace_module

    def test_forward(
        self, model: Union[LaplaceRegression, LaplaceClassification]
    ) -> None:
        """Test forward pass of Laplace model."""
        n_inputs = model.num_input_dims
        n_outputs = model.num_output_dims
        X = torch.randn(5, n_inputs)
        out = model(X)

    def test_predict_step(
        self, model: Union[LaplaceRegression, LaplaceClassification]
    ) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5
