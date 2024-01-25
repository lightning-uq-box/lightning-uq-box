# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Image Diffusion Models Tasks."""

import builtins
from pathlib import Path
from typing import Any

import pytest
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from pytest import MonkeyPatch

model_config_paths = [
    "tests/configs/diffusion_models/guided_diffusion.yaml",
    "tests/configs/diffusion_models/guided_diffusion_classifier.yaml",
]

data_config_paths = ["tests/configs/image_classification/toy_classification.yaml"]

trainer_config_path = ["tests/configs/diffusion_models/trainer.yaml"]


class TestImageClassificationTask:
    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "guided_diffusion":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    @pytest.mark.parametrize("trainer_config_path", trainer_config_path)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        trainer_config_path: str,
        tmp_path: Path,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        trainer_conf = OmegaConf.load(trainer_config_path)

        full_conf = OmegaConf.merge(trainer_conf, data_conf, model_conf)

        if "classifier" in model_config_path:
            # provide unconditional diffusion model
            from lightning_uq_box.uq_methods.guided_diffusion_model import (
                my_create_model,
            )

            uncond_model = my_create_model(
                image_size=64,
                num_channels=128,
                num_res_blocks=2,
                channel_mult="",
                learn_sigma=False,
                use_checkpoint=False,
                attention_resolutions="16,8",
                num_heads=4,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                dropout=0.0,
                resblock_updown=False,
                use_fp16=False,
                use_new_attention_order=False,
                num_classes=4,
            )
            model = instantiate(full_conf.model, model=uncond_model)
        else:
            model = instantiate(full_conf.model)

        datamodule = instantiate(full_conf.data)
        trainer = instantiate(
            full_conf.trainer,
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )

        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)

        # test predict step with a single batch input
        batch = next(iter(datamodule.val_dataloader()))
        pred_dict = model.predict_step(batch["input"])

        assert "sample" in pred_dict
