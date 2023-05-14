"""Unit Test SGLD Method."""


import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.uq_methods import SGLDModel


class TestSGLDModel:
    @pytest.fixture(params=["sgld_nll.yaml", "sgld_mse.yaml"])
    def sgld_model(self, tmp_path: Path, request: SubRequest) -> SGLDModel:
        """Create a base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", request.param))
        conf.uq_method["save_dir"] = str(tmp_path)
        dm = ToyHeteroscedasticDatamodule()
        return instantiate(conf.uq_method, train_loader=dm.train_dataloader())

    def test_predict_step(self, sgld_model: SGLDModel) -> None:
        """Test forward pass of base model."""
        n_inputs = sgld_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = sgld_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, sgld_model: SGLDModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=5,
            default_root_dir=sgld_model.hparams.save_dir,
        )
        trainer.test(model=sgld_model, datamodule=datamodule)
