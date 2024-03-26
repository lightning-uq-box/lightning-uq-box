# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Test pixelwise regression task."""

import os
from pathlib import Path

import h5py
import pytest
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

seed_everything(0)

model_config_paths = [
    "tests/configs/pixelwise_regression/base.yaml",
    "tests/configs/pixelwise_regression/mve.yaml",
    "tests/configs/pixelwise_regression/der.yaml",
    "tests/configs/pixelwise_regression/quantile_regression.yaml",
    "tests/configs/pixelwise_regression/img2img_conformal.yaml",
    "tests/configs/pixelwise_regression/img2img_conformal_torchseg.yaml",
    "tests/configs/pixelwise_regression/mc_dropout.yaml",
]

data_config_paths = ["tests/configs/pixelwise_regression/toy_pixelwise_regression.yaml"]


class TestImageClassificationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            max_epochs=2,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
            deterministic=True,
            logger=CSVLogger(str(tmp_path)),
        )

        if "conformal" in model_config_path:
            trainer.validate(model, datamodule.calib_dataloader())
        else:
            trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        with h5py.File(os.path.join(model.pred_dir, "batch_0_sample_0.hdf5"), "r") as f:
            assert "pred" in f
            assert "target" in f
            assert "aux" in f.attrs
            assert "index" in f.attrs


frozen_config_paths = [
    "tests/configs/pixelwise_regression/base.yaml",
    "tests/configs/pixelwise_regression/mc_dropout.yaml",
    "tests/configs/pixelwise_regression/quantile_regression.yaml",
    "tests/configs/pixelwise_regression/mve.yaml",
    "tests/configs/pixelwise_regression/der.yaml",
]


class TestFrozenSegmentation:
    @pytest.mark.parametrize("model_name", ["Unet", "DeepLabV3Plus"])
    @pytest.mark.parametrize("backbone", ["resnet18", "vit_tiny_patch16_224"])
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(
        self, model_config_path: str, model_name: str, backbone: str
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        model_conf.uq_method.model["_target_"] = f"torchseg.{model_name}"
        model_conf.uq_method.model["encoder_name"] = backbone

        module = instantiate(model_conf.uq_method, freeze_backbone=True)
        seg_model = module.model

        assert all(
            [param.requires_grad is False for param in seg_model.encoder.parameters()]
        )
        assert all([param.requires_grad for param in seg_model.decoder.parameters()])
        assert all(
            [param.requires_grad for param in seg_model.segmentation_head.parameters()]
        )

    @pytest.mark.parametrize("model_name", ["Unet", "DeepLabV3Plus"])
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_decoder(self, model_config_path: str, model_name: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        model_conf.uq_method.model["_target_"] = f"torchseg.{model_name}"

        module = instantiate(model_conf.uq_method, freeze_decoder=True)
        seg_model = module.model

        assert all(
            [param.requires_grad is False for param in seg_model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in seg_model.encoder.parameters()])
        assert all(
            [param.requires_grad for param in seg_model.segmentation_head.parameters()]
        )
