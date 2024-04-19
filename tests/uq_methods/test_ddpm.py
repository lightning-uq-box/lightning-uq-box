# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Image Diffusion Models Tasks."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch import Tensor

model_config_paths = [
    "tests/configs/diffusion_models/ddpm.yaml",
    # "tests/configs/diffusion_models/guided_ddpm.yaml",
    # "tests/configs/diffusion_models/guidance_free_ddpm.yaml",
]

data_config_paths = ["tests/configs/image_classification/toy_classification.yaml"]

trainer_config_path = ["tests/configs/diffusion_models/trainer.yaml"]


class TestDDPMTasks:
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
        if "guidance_free" in model_config_path:
            sampled_images = model(batch["target"])
        else:
            sampled_images = model(batch["input"].shape[0])

        assert isinstance(sampled_images, Tensor)
