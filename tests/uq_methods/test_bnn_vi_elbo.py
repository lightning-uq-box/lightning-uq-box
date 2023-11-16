"""Unit test Bayes By Backprop Implementation."""

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

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.uq_methods import (
    BNN_VI_ELBO_Classification,
    BNN_VI_ELBO_Regression,
)


class TestBNN_VI_ELBO:
    @pytest.fixture(params=["bnn_vi_elbo_regression.yaml"])
    def model_regression(self, request: SubRequest) -> BNN_VI_ELBO_Regression:
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "bnn_vi_elbo", request.param)
        )
        return instantiate(conf.uq_method)

    @pytest.fixture(params=["bnn_vi_elbo_classification.yaml"])
    def model_classification(self, request: SubRequest) -> BNN_VI_ELBO_Regression:
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "bnn_vi_elbo", request.param)
        )
        return instantiate(conf.uq_method)

    @pytest.mark.parametrize(
        "model",
        [lazy_fixture("model_regression"), lazy_fixture("model_classification")],
    )
    def test_forward(
        self, model: Union[BNN_VI_ELBO_Regression, BNN_VI_ELBO_Classification]
    ) -> None:
        """Test forward pass of base model."""
        n_inputs = model.num_input_dims
        n_outputs = model.num_output_dims
        X = torch.randn(5, n_inputs)
        out = model(X)
        assert out.shape[-1] == n_outputs

    @pytest.mark.parametrize(
        "model",
        [lazy_fixture("model_regression"), lazy_fixture("model_classification")],
    )
    def test_predict_step(
        self, model: Union[BNN_VI_ELBO_Regression, BNN_VI_ELBO_Classification]
    ) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = model.num_input_dims
        X = torch.randn(5, n_inputs)
        out = model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], torch.Tensor)
        assert out["pred"].shape[0] == 5

    @pytest.mark.parametrize(
        "model, datamodule",
        [
            (lazy_fixture("model_regression"), ToyHeteroscedasticDatamodule()),
            (lazy_fixture("model_classification"), TwoMoonsDataModule()),
        ],
    )
    def test_trainer(self, model, datamodule, tmp_path: Path) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
