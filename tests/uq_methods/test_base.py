"""Unit tests for Base Model."""

import os
import sys
from pathlib import Path

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
from uq_method_box.uq_methods import BaseModel


class TestBaseModel:
    @pytest.fixture
    def base_model(self, tmp_path: Path) -> BaseModel:
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

    def test_forward(self, base_model: BaseModel) -> None:
        """Test forward pass of base model."""
        n_inputs = base_model.hparams.model_args["n_inputs"]
        n_outputs = base_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = base_model(X)
        assert out.shape[-1] == n_outputs

    def test_trainer(self, base_model: BaseModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=base_model.hparams.save_dir,
        )
        trainer.fit(model=base_model, datamodule=datamodule)
        trainer.test(model=base_model, datamodule=datamodule)
