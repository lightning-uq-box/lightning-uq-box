# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.
"""Test image segmentation task."""

import glob
from pathlib import Path
from typing import Any, Dict

import pytest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToySegmentationDataModule
from lightning_uq_box.uq_methods import DeepEnsembleSegmentation

model_config_paths = [
    "tests/configs/image_segmentation/bnn_vi_elbo.yaml",
    "tests/configs/image_segmentation/bnn_vi_elbo_part_stoch.yaml",
    "tests/configs/image_segmentation/mc_dropout.yaml",
    "tests/configs/image_segmentation/swag.yaml",
    "tests/configs/image_segmentation/prob_unet.yaml",
]

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


ensemble_model_config_paths = ["tests/configs/image_segmentation/mc_dropout.yaml"]


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
        for i in range(5):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.model)
            datamodule = instantiate(data_conf.data)
            trainer = Trainer(
                max_epochs=2, log_every_n_steps=1, default_root_dir=str(tmp_path)
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
        self, ensemble_members_dict: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleSegmentation(
            len(ensemble_members_dict), ensemble_members_dict, num_classes=4
        )

        datamodule = ToySegmentationDataModule()

        trainer = Trainer(default_root_dir=str(tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)
