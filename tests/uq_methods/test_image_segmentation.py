# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Test image segmentation task."""

import glob
import os
from pathlib import Path
from typing import Any

import h5py
import pytest
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToySegmentationDataModule
from lightning_uq_box.uq_methods import DeepEnsembleSegmentation

seed_everything(0)

model_config_paths = [
    "tests/configs/image_segmentation/base.yaml",
    "tests/configs/image_segmentation/bnn_vi_elbo.yaml",
    "tests/configs/image_segmentation/bnn_vi_elbo_part_stoch.yaml",
    "tests/configs/image_segmentation/mc_dropout.yaml",
    "tests/configs/image_segmentation/swag.yaml",
    "tests/configs/image_segmentation/prob_unet.yaml",
    "tests/configs/image_segmentation/hierarchical_prob_unet.yaml",
]

data_config_paths = ["tests/configs/image_segmentation/toy_segmentation.yaml"]


class TestImageSegmentationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method, save_preds=True)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=2,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
            deterministic=True,
            logger=CSVLogger(str(tmp_path)),
        )

        if "mc_dropout" in model_config_path:
            with pytest.raises(UserWarning, match="No dropout layers found in model"):
                trainer.fit(model, datamodule)
                trainer.test(ckpt_path="best", datamodule=datamodule)
        else:
            trainer.fit(model, datamodule)
            trainer.test(ckpt_path="best", datamodule=datamodule)

        with h5py.File(os.path.join(model.pred_dir, "batch_0_sample_0.hdf5"), "r") as f:
            assert "pred" in f
            assert "target" in f
            for key, value in f.items():
                assert value.shape[-1] == 64
                assert value.shape[-2] == 64
            assert "aux" in f.attrs
            assert "index" in f.attrs


ensemble_model_config_paths = ["tests/configs/image_segmentation/deterministic.yaml"]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(self, request, tmp_path_factory: TempPathFactory) -> None:
        model_config_path, data_config_path = request.param
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(3):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.uq_method, save_preds=True)
            datamodule = instantiate(data_conf.data)
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=1,
                log_every_n_steps=1,
                default_root_dir=str(tmp_path),
            )
            trainer.fit(model, datamodule)
            trainer.test(ckpt_path="best", datamodule=datamodule)

            # Find the .ckpt file in the lightning_logs directory
            ckpt_file = glob.glob(
                f"{str(tmp_path)}/lightning_logs/version_*/checkpoints/*.ckpt"
            )[0]
            ckpt_paths.append({"base_model": model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self, ensemble_members_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleSegmentation(
            ensemble_members_dict, num_classes=4, save_preds=True
        )

        datamodule = ToySegmentationDataModule()

        trainer = Trainer(accelerator="cpu", default_root_dir=str(tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)

        with h5py.File(
            os.path.join(ensemble_model.pred_dir, "batch_0_sample_0.hdf5"), "r"
        ) as f:
            assert "pred" in f
            assert "target" in f
            for key, value in f.items():
                assert value.shape[-1] == 64
                assert value.shape[-2] == 64
            assert "aux" in f.attrs
            assert "index" in f.attrs


frozen_config_paths = [
    "tests/configs/image_segmentation/base.yaml",
    "tests/configs/image_segmentation/mc_dropout.yaml",
    "tests/configs/image_segmentation/bnn_vi_elbo.yaml",
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
