# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Test image segmentation task."""

import os
from pathlib import Path
from typing import Any

import h5py
import pytest
from conftest import minimal_trainer_kwargs
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch import seed_everything
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
]

data_config_paths = ["tests/configs/image_segmentation/toy_segmentation.yaml"]


class TestImageSegmentationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        tmp_path: Path,
        accelerator_config,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method, save_preds=True)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            **minimal_trainer_kwargs(
                accelerator_config,
                tmp_path,
                max_epochs=2 if "swag" in model_config_path else 1,
                checkpoints=True,
            )
        )

        trainer.fit(model, datamodule)
        if "mc_dropout" in model_config_path:
            with pytest.raises(UserWarning, match="No dropout layers found in model"):
                trainer.test(ckpt_path="best", datamodule=datamodule)
        else:
            trainer.test(ckpt_path="best", datamodule=datamodule)

            with h5py.File(
                os.path.join(model.pred_dir, "batch_0_sample_0.hdf5"), "r"
            ) as f:
                assert "pred" in f
                assert "target" in f
                for key, value in f.items():
                    if key == "logits":
                        assert value.shape[1] == 64
                        assert value.shape[2] == 64
                    else:
                        assert value.shape[-1] == 64
                        assert value.shape[-2] == 64
                assert "aux" in f.attrs
                assert "index" in f.attrs


ensemble_model_config_paths = ["tests/configs/image_segmentation/base.yaml"]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(
        self, request, tmp_path_factory: TempPathFactory, accelerator_config
    ) -> list[dict[str, Any]]:
        model_config_path, data_config_path = request.param
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(2):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.uq_method, save_preds=True)
            datamodule = instantiate(data_conf.data)
            trainer = Trainer(
                **minimal_trainer_kwargs(accelerator_config, tmp_path, checkpoints=True)
            )
            trainer.fit(model, datamodule)
            ckpt_file = trainer.checkpoint_callback.best_model_path
            assert ckpt_file
            ckpt_paths.append({"base_model": model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self,
        ensemble_members_dict: list[dict[str, Any]],
        tmp_path: Path,
        accelerator_config,
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleSegmentation(
            ensemble_members_dict, num_classes=4, save_preds=True
        )

        datamodule = ToySegmentationDataModule(num_images=2, batch_size=2)

        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)

        with h5py.File(
            os.path.join(ensemble_model.pred_dir, "batch_0_sample_0.hdf5"), "r"
        ) as f:
            assert "pred" in f
            assert "target" in f
            for key, value in f.items():
                if key == "logits":
                    assert value.shape[1] == 64
                    assert value.shape[2] == 64
                else:
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
    @pytest.mark.parametrize(
        "backbone", ["resnet18", "tu-swin_tiny_patch4_window7_224"]
    )
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(
        self, model_config_path: str, model_name: str, backbone: str
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        model_conf.uq_method.model["_target_"] = (
            f"segmentation_models_pytorch.{model_name}"
        )
        model_conf.uq_method.model["encoder_name"] = backbone

        if model_name == "DeepLabV3Plus":
            # drop depth and decoder_channels
            model_conf.uq_method.model.pop("encoder_depth")
            model_conf.uq_method.model.pop("decoder_channels")

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
        model_conf.uq_method.model["_target_"] = (
            f"segmentation_models_pytorch.{model_name}"
        )

        if model_name == "DeepLabV3Plus":
            # drop depth and decoder_channels as this decoder needs different config
            model_conf.uq_method.model.pop("encoder_depth")
            model_conf.uq_method.model.pop("decoder_channels")

        module = instantiate(model_conf.uq_method, freeze_decoder=True)
        seg_model = module.model

        assert all(
            [param.requires_grad is False for param in seg_model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in seg_model.encoder.parameters()])
        assert all(
            [param.requires_grad for param in seg_model.segmentation_head.parameters()]
        )
