"""Test Determinist Gaussian Model."""

import os
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageRegressionDatamodule,
)
from lightning_uq_box.uq_methods import BaseModel


class TestBaseModel:
    @pytest.fixture
    def base_model_tabular(self, tmp_path: Path) -> BaseModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "base.yaml"))
        conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_forward_tabular(self, base_model_tabular: BaseModel) -> None:
        """Test forward pass of base model."""
        n_inputs = base_model_tabular.num_inputs
        n_outputs = base_model_tabular.num_outputs
        X = torch.randn(5, n_inputs)
        out = base_model_tabular(X)
        assert out.shape[-1] == n_outputs

    def test_trainer_tabular(self, base_model_tabular: BaseModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=base_model_tabular.hparams.save_dir,
        )
        trainer.fit(model=base_model_tabular, datamodule=datamodule)
        trainer.test(model=base_model_tabular, datamodule=datamodule)

    @pytest.fixture(
        params=[("resnet18", "timm_config.yaml")]
    )  # , ("swinv2_tiny_window8_256", "timm_config.yaml")])
    def base_model_timm(self, tmp_path: Path, request: SubRequest) -> BaseModel:
        _, filename = request.param
        conf = OmegaConf.load(os.path.join("tests", "configs", filename))
        conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_forward_image(self, base_model_timm: BaseModel) -> None:
        """Test forward pass of base model."""
        n_inputs = base_model_timm.num_inputs
        n_outputs = base_model_timm.num_outputs
        X = torch.randn(2, n_inputs, 256, 256)
        out = base_model_timm(X)
        assert out.shape[-1] == n_outputs

    def test_trainer_image(self, base_model_timm: BaseModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=base_model_timm.hparams.save_dir,
        )
        trainer.fit(model=base_model_timm, datamodule=datamodule)
        trainer.test(model=base_model_timm, datamodule=datamodule)
