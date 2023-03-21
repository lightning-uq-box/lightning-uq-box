"""Test Determinist Gaussian Model."""

import os
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageRegressionDatamodule,
)
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BaseModel


class TestBaseModel:
    @pytest.fixture
    def base_model_tabular(self, tmp_path: Path) -> BaseModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "base.yaml"))
        conf_dict = OmegaConf.to_object(conf)
        return BaseModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            loss_fn="mse",
            save_dir=tmp_path,
        )

    def test_forward_tabular(self, base_model_tabular: BaseModel) -> None:
        """Test forward pass of base model."""
        n_inputs = base_model_tabular.hparams.model_args["n_inputs"]
        n_outputs = base_model_tabular.hparams.model_args["n_outputs"]
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
        backbone, filename = request.param
        conf = OmegaConf.load(os.path.join("tests", "configs", filename))
        conf_dict = OmegaConf.to_object(conf)
        return BaseModel(
            backbone,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            loss_fn="mse",
            save_dir=tmp_path,
        )

    def test_forward_image(self, base_model_timm: BaseModel) -> None:
        """Test forward pass of base model."""
        n_inputs = base_model_timm.hparams.model_args["in_chans"]
        n_outputs = base_model_timm.hparams.model_args["num_classes"]
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
