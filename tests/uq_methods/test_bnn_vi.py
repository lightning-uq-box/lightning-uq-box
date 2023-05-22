"""Test BNN with VI."""

import os
from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from uq_method_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageRegressionDatamodule,
)
from uq_method_box.uq_methods import BNN_VI, BNN_VI_Batched


class TestBNN_VI_Model:
    # test for tabular data
    @pytest.fixture(
        params=product(
            [
                "uq_method_box.uq_methods.BNN_VI",
                "uq_method_box.uq_methods.BNN_VI_Batched",
            ],
            ["reparameterization", "flipout"],
            [None, [-1], ["model.0"]],
        )  # test everything for both layer_types
    )
    def bnn_vi_model_tabular(
        self, tmp_path: Path, request: SubRequest
    ) -> Union[BNN_VI, BNN_VI_Batched]:
        """Create BNN_VI model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "bnn_vi.yaml"))
        dm = ToyHeteroscedasticDatamodule()
        conf.uq_method["_target_"] = request.param[0]
        conf.uq_method["save_dir"] = str(tmp_path)
        conf.uq_method["num_training_points"] = dm.X_train.shape[0]
        conf.uq_method["layer_type"] = request.param[1]
        conf.uq_method["part_stoch_module_names"] = request.param[2]
        return instantiate(conf.uq_method)

    # tests for tabular data
    def test_forward(self, bnn_vi_model_tabular: Union[BNN_VI, BNN_VI_Batched]) -> None:
        """Test forward pass of model."""
        n_inputs = bnn_vi_model_tabular.num_inputs
        X = torch.randn(4, n_inputs)
        out = bnn_vi_model_tabular(X)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 4
        assert out.shape[1] == 1

    def test_predict_step(
        self, bnn_vi_model_tabular: Union[BNN_VI, BNN_VI_Batched]
    ) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = bnn_vi_model_tabular.num_inputs
        X = torch.randn(4, n_inputs)
        out = bnn_vi_model_tabular.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 4

    def test_trainer(self, bnn_vi_model_tabular: Union[BNN_VI, BNN_VI_Batched]) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            logger=False,
            max_epochs=1,
            default_root_dir=bnn_vi_model_tabular.hparams.save_dir,
        )
        trainer.test(model=bnn_vi_model_tabular, datamodule=datamodule)

    # test for image task
    @pytest.fixture(
        params=product(
            ["reparameterization", "flipout"],
            [[-1], ["layer4.1.conv1", "layer4.1.conv2"]],
        )  # test everything for both layer_types
    )
    def bnn_vi_model_image(self, tmp_path: Path, request: SubRequest) -> BNN_VI:
        """Create BNN_VI model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "bnn_vi.yaml"))
        dm = ToyHeteroscedasticDatamodule()
        conf.uq_method["save_dir"] = str(tmp_path)
        conf.uq_method["num_training_points"] = dm.X_train.shape[0]
        conf.uq_method["layer_type"] = request.param[0]
        conf.uq_method["part_stoch_module_names"] = request.param[1]
        model = timm.create_model("resnet18", in_chans=3, num_classes=1)
        return instantiate(conf.uq_method, model=model)

    # tests for image data
    def test_forward_image(self, bnn_vi_model_image: BNN_VI) -> None:
        """Test forward pass of model."""
        X = torch.randn(2, 3, 32, 32)
        out = bnn_vi_model_image(X)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 2
        assert out.shape[-1] == 1

    def test_predict_step_image(self, bnn_vi_model_image: BNN_VI) -> None:
        """Test predict step outside of Lightning Trainer."""
        X = torch.randn(2, 3, 32, 32)
        out = bnn_vi_model_image.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 2

    def test_trainer_image(self, bnn_vi_model_image: BNN_VI) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(
            logger=False,
            max_epochs=1,
            default_root_dir=bnn_vi_model_image.hparams.save_dir,
        )
        trainer.test(model=bnn_vi_model_image, datamodule=datamodule)
