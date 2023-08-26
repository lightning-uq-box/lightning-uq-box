"""Test MC-Dropout Model."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import MCDropoutModel


# TODO test different both mse and nll
class TestMCDropoutModel:
    @pytest.fixture(params=["mc_dropout_nll.yaml", "mc_dropout_mse.yaml"])
    def mc_model(self, tmp_path: Path, request: SubRequest) -> MCDropoutModel:
        """Create a MC Dropout model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", request.param))
        conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_forward(self, mc_model: MCDropoutModel) -> None:
        """Test forward pass of MC dropout model."""
        n_inputs = mc_model.num_inputs
        n_outputs = mc_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = mc_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, mc_model: MCDropoutModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = mc_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = mc_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_trainer(self, mc_model: MCDropoutModel) -> None:
        """Test MC Dropout Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=2,
            default_root_dir=mc_model.hparams.save_dir,
        )
        trainer.fit(model=mc_model, datamodule=datamodule)
        trainer.test(mc_model, datamodule.test_dataloader())
