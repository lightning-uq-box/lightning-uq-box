"""Test BNN with VI."""

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

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.uq_methods import BNN_VI


class TestBNN_VI_Model:
    @pytest.fixture(
        params=["reparameterization", "flipout"]  # test everything for both layer_types
    )
    def bnn_vi_model(self, tmp_path: Path, request: SubRequest) -> BNN_VI:
        """Create BNN_VI model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "bnn_vi.yaml"))
        dm = ToyHeteroscedasticDatamodule()
        conf.uq_method["save_dir"] = str(tmp_path)
        conf.uq_method["num_training_points"] = dm.X_train.shape[0]
        conf.uq_method["layer_type"] = request.param
        return instantiate(conf.uq_method)

    def test_forward(self, bnn_vi_model: BNN_VI) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = bnn_vi_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = bnn_vi_model(X)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 5
        assert out.shape[-1] == 1

    def test_predict_step(self, bnn_vi_model: BNN_VI) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = bnn_vi_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = bnn_vi_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, bnn_vi_model: BNN_VI) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=bnn_vi_model.hparams.save_dir,
        )
        trainer.test(model=bnn_vi_model, datamodule=datamodule)
