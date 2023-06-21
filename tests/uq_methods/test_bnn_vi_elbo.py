"""Unit test Bayes By Backprop Implementation."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.uq_methods import BNN_VI_ELBO


class TestBNN_VI_ELBO:
    @pytest.fixture
    def bnn_vi_elbo_model(self, tmp_path: Path) -> BNN_VI_ELBO:
        conf = OmegaConf.load(os.path.join("tests", "configs", "bnn_vi_elbo.yaml"))
        conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_forward(self, bnn_vi_elbo_model: BNN_VI_ELBO) -> None:
        """Test forward pass of base model."""
        n_inputs = bnn_vi_elbo_model.num_inputs
        n_outputs = bnn_vi_elbo_model.num_outputs
        X = torch.randn(5, n_inputs)
        out = bnn_vi_elbo_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, bnn_vi_elbo_model: BNN_VI_ELBO) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = bnn_vi_elbo_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = bnn_vi_elbo_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, bnn_vi_elbo_model: BNN_VI_ELBO) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=bnn_vi_elbo_model.hparams.save_dir,
        )
        trainer.fit(model=bnn_vi_elbo_model, datamodule=datamodule)
        trainer.test(model=bnn_vi_elbo_model, datamodule=datamodule)
