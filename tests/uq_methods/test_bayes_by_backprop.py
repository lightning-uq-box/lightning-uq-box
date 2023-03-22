"""Unit test Bayes By Backprop Implementation."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BayesByBackpropModel


class TestBayesByBackpropModel:
    @pytest.fixture
    def bayes_by_backprop_model(self, tmp_path: Path) -> BayesByBackpropModel:
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "bayes_by_backprop.yaml")
        )
        conf_dict = OmegaConf.to_object(conf)
        return BayesByBackpropModel(
            MLP,
            model_args=conf_dict["model"]["model_args"],
            lr=1e-3,
            loss_fn="mse",
            save_dir=tmp_path,
            **conf_dict["model"]["bayes_by_backprop"],
        )

    def test_forward(self, bayes_by_backprop_model: BayesByBackpropModel) -> None:
        """Test forward pass of base model."""
        n_inputs = bayes_by_backprop_model.hparams.model_args["n_inputs"]
        n_outputs = bayes_by_backprop_model.hparams.model_args["n_outputs"]
        X = torch.randn(5, n_inputs)
        out = bayes_by_backprop_model(X)
        assert out.shape[-1] == n_outputs

    def test_predict_step(self, bayes_by_backprop_model: BayesByBackpropModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = bayes_by_backprop_model.hparams.model_args["n_inputs"]
        X = torch.randn(5, n_inputs)
        out = bayes_by_backprop_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, bayes_by_backprop_model: BayesByBackpropModel) -> None:
        """Test Base Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=bayes_by_backprop_model.hparams.save_dir,
        )
        trainer.fit(model=bayes_by_backprop_model, datamodule=datamodule)
        trainer.test(model=bayes_by_backprop_model, datamodule=datamodule)
