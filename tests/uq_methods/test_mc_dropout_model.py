"""Test MC-Dropout Model."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import MCDropoutModel


# TODO test different both mse and nll
class TestMCDropoutModel:
    @pytest.fixture(params=["mc_dropout_mse.yaml", "mc_dropout_nll.yaml"])
    def mc_model(self, tmp_path: Path, request: SubRequest) -> MCDropoutModel:
        """Create a QR model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", request.param))
        conf_dict = OmegaConf.to_object(conf)
        return MCDropoutModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            num_mc_samples=conf_dict["model"]["num_mc_samples"],
            lr=1e-3,
            loss_fn=conf_dict["model"]["loss_fn"],
            save_dir=tmp_path,
        )

    def test_forward(self, mc_model: MCDropoutModel) -> None:
        """Test forward pass of QR model."""
        n_inputs = mc_model.hparams.model_args["n_inputs"]
        n_outputs = mc_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = mc_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, mc_model: MCDropoutModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = mc_model.hparams.model_args["n_inputs"]
        X = torch.randn(5, n_inputs)
        out = mc_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, mc_model: MCDropoutModel) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=mc_model.hparams.save_dir,
        )
        trainer.fit(model=mc_model, datamodule=datamodule)
        trainer.test(mc_model, datamodule.test_dataloader())
