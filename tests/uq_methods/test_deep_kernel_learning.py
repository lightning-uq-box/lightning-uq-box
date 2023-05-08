"""Test Deep Kernel Learning Model."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.uq_methods import DeepKernelLearningModel

# TODO need to test all different laplace args


class TestDeepKernelLearningModel:
    @pytest.fixture
    def dkl_model(self, tmp_path: Path) -> DeepKernelLearningModel:
        """Create DKL model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "deep_kernel_learning.yaml")
        )
        conf.uq_method["save_dir"] = str(tmp_path)
        datamodule = ToyHeteroscedasticDatamodule()
        train_loader = datamodule.train_dataloader()
        return instantiate(conf.uq_method, train_loader=train_loader)

    def test_forward(self, dkl_model: DeepKernelLearningModel) -> None:
        """Test forward pass of conformalized model."""
        n_inputs = dkl_model.num_inputs
        X = torch.randn(5, n_inputs)
        out = dkl_model(X)
        assert isinstance(out, MultivariateNormal)

    def test_predict_step(self, dkl_model: DeepKernelLearningModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = dkl_model.num_inputs
        X = torch.randn(5, n_inputs)
        # backpack expects a torch.nn.sequential but also works otherwise
        out = dkl_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_trainer(self, dkl_model: DeepKernelLearningModel) -> None:
        """Test QR Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=dkl_model.hparams.save_dir,
        )
        # backpack expects a torch.nn.sequential but also works otherwise
        trainer.test(model=dkl_model, datamodule=datamodule)
