# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.
"""Test image segmentation task."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

model_config_paths = ["tests/configs/image_segmentation/mc_dropout.yaml"]

data_config_paths = ["tests/configs/image_segmentation/toy_segmentation.yaml"]


class TestImageClassificationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.model)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            max_epochs=2, log_every_n_steps=1, default_root_dir=str(tmp_path)
        )

        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)
