"""Test SGLD Model."""
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
from lightning_uq_box.uq_methods import SGLDClassification, SGLDRegression


class TestMCDropout:
    @pytest.fixture(params=["sgld_nll.yaml", "sgld_mse.yaml"])
    def model_regression(self, request: SubRequest) -> SGLDRegression:
        """Create a SGLD Regression Model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "sgld", request.param))
        return instantiate(conf.uq_method)

    @pytest.fixture(params=["sgld_class.yaml"])
    def model_classification(self, request: SubRequest) -> SGLDClassification:
        """Create a SGLD Classification Model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "sgld", request.param))
        return instantiate(conf.uq_method)

    @pytest.mark.parametrize(
        "model, datamodule",
        [
            (lazy_fixture("model_regression"), ToyHeteroscedasticDatamodule()),
            (lazy_fixture("model_classification"), TwoMoonsDataModule()),
        ],
    )
    def test_trainer(self, model, datamodule, tmp_path: Path) -> None:
        """Test MC Dropout Model with a Lightning Trainer."""
        # instantiate datamodule
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=2, default_root_dir=str(tmp_path)
        )
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model, datamodule.test_dataloader())
