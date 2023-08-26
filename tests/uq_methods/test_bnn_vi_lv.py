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

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageRegressionDatamodule,
)
from lightning_uq_box.uq_methods import BNN_LV_VI, BNN_LV_VI_Batched


class TestBNN_LV_VI_Model:
    # fixture for iterative sampling
    @pytest.fixture(
        params=product(
            [
                "lightning_uq_box.uq_methods.BNN_LV_VI",
                "lightning_uq_box.uq_methods.BNN_LV_VI_Batched",
            ],
            ["reparameterization", "flipout"],  # layer types
            ["first", "last"],  # LV intro options
            [None, [-1], ["model.0"]],  # part stochastic
        )
    )
    def bnn_vi_lv_model_tabular(self, tmp_path: Path, request: SubRequest) -> BNN_LV_VI:
        """Create BNN_LV_VI model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", f"bnn_vi_lv_{request.param[2]}.yaml")
        )
        dm = ToyHeteroscedasticDatamodule()
        conf.uq_method["_target_"] = request.param[0]
        conf.uq_method["save_dir"] = str(tmp_path)
        conf.uq_method["num_training_points"] = dm.X_train.shape[0]
        conf.uq_method["layer_type"] = request.param[1]
        conf.uq_method["latent_variable_intro"] = request.param[2]
        conf.uq_method["part_stoch_module_names"] = request.param[3]
        return instantiate(conf.uq_method)

    # tests for tabular data
    def test_forward(
        self, bnn_vi_lv_model_tabular: Union[BNN_LV_VI, BNN_LV_VI_Batched]
    ) -> None:
        """Test forward pass of model."""
        X = torch.randn(3, 1)
        y = torch.randn(3, 1)
        out = bnn_vi_lv_model_tabular(X, y)
        assert isinstance(out, Tensor)
        assert out.shape[-2] == 3
        assert out.shape[-1] == 1

    def test_predict_step(
        self, bnn_vi_lv_model_tabular: Union[BNN_LV_VI, BNN_LV_VI_Batched]
    ) -> None:
        """Test predict step outside of Lightning Trainer."""
        X = torch.randn(3, 1)
        out = bnn_vi_lv_model_tabular.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 3

    def test_trainer(
        self, bnn_vi_lv_model_tabular: Union[BNN_LV_VI, BNN_LV_VI_Batched]
    ) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            logger=False,
            max_epochs=1,
            default_root_dir=bnn_vi_lv_model_tabular.hparams.save_dir,
        )
        trainer.test(model=bnn_vi_lv_model_tabular, datamodule=datamodule)

    # # tests for image data
    @pytest.fixture(
        params=product(
            ["reparameterization", "flipout"],  # layer types
            [None, [-1], ["layer4.1.conv1", "layer4.1.conv2"]],  # part stochastic
        )
    )
    def bnn_vi_lv_model_image(self, tmp_path: Path, request: SubRequest) -> BNN_LV_VI:
        """Create BNN_LV_VI model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "bnn_vi_lv_last.yaml"))
        dm = ToyHeteroscedasticDatamodule()

        conf.uq_method["save_dir"] = str(tmp_path)
        conf.uq_method["num_training_points"] = dm.X_train.shape[0]
        conf.uq_method["layer_type"] = request.param[0]
        conf.uq_method["latent_variable_intro"] = "last"
        conf.uq_method["part_stoch_module_names"] = request.param[1]
        conf.uq_method["latent_net"][
            "n_inputs"
        ] = 513  # resnet18 hast 512 output feature dim + 1 target dim noqa: E501
        model = timm.create_model("resnet18", in_chans=3, num_classes=1)
        return instantiate(conf.uq_method, model=model)

    def test_forward_image(self, bnn_vi_lv_model_image: BNN_LV_VI) -> None:
        """Test forward pass of model."""
        X = torch.randn(2, 3, 32, 32)
        y = torch.randn(2, 1)
        out = bnn_vi_lv_model_image(X=X, y=y)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 2
        assert out.shape[-1] == 1

    def test_predict_step_image(self, bnn_vi_lv_model_image: BNN_LV_VI) -> None:
        """Test predict step outside of Lightning Trainer."""
        X = torch.randn(2, 3, 32, 32)
        out = bnn_vi_lv_model_image.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 2

    def test_trainer_image(self, bnn_vi_lv_model_image: BNN_LV_VI) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(
            logger=False,
            max_epochs=1,
            default_root_dir=bnn_vi_lv_model_image.hparams.save_dir,
        )
        trainer.test(model=bnn_vi_lv_model_image, datamodule=datamodule)
