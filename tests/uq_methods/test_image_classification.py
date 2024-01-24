# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Image Classification Tasks."""

import glob
from pathlib import Path
from typing import Any, Dict

import pytest
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyImageClassificationDatamodule
from lightning_uq_box.uq_methods import DeepEnsembleClassification, TTAClassification

model_config_paths = [
    "tests/configs/image_classification/mc_dropout.yaml",
    "tests/configs/image_classification/bnn_vi_elbo.yaml",
    "tests/configs/image_classification/swag.yaml",
    "tests/configs/image_classification/sgld.yaml",
    "tests/configs/image_classification/dkl.yaml",
    "tests/configs/image_classification/due.yaml",
    "tests/configs/image_classification/laplace.yaml",
    "tests/configs/image_classification/card.yaml",
]

data_config_paths = ["tests/configs/image_classification/toy_classification.yaml"]


class TestImageClassificationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        # timm resnets implement dropout as nn.functional and not modules
        # so the find_dropout_layers function yields a warning
        # TODO
        # match = "No dropout layers found in model*"
        # with pytest.warns(UserWarning):
        model = instantiate(model_conf.model)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            max_epochs=2,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )
        # laplace only uses test
        if "laplace" not in model_config_path:
            trainer.fit(model, datamodule)
            trainer.test(ckpt_path="best", datamodule=datamodule)
        else:
            trainer.test(model, datamodule=datamodule)


posthoc_config_paths = [
    "tests/configs/image_classification/temp_scaling.yaml",
    "tests/configs/image_classification/raps.yaml",
    "tests/configs/image_classification/raps_bnn.yaml",
]


class TestPosthoc:
    @pytest.mark.parametrize("model_config_path", posthoc_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.model)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(default_root_dir=str(tmp_path), inference_mode=False)
        # use validation for testing, should be calibration loader for conformal
        trainer.validate(model, datamodule.val_dataloader())
        trainer.test(model, datamodule=datamodule)


ensemble_model_config_paths = ["tests/configs/image_classification/mc_dropout.yaml"]


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
        ensemble_model = DeepEnsembleClassification(
            len(ensemble_members_dict), ensemble_members_dict, num_classes=4
        )

        datamodule = ToyImageClassificationDatamodule()

        trainer = Trainer(default_root_dir=str(tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)


tta_model_paths = [
    "tests/configs/image_classification/mc_dropout.yaml",
    "tests/configs/image_classification/tta_augmentation.yaml",
]


class TestTTAModel:
    @pytest.mark.parametrize("model_config_path", tta_model_paths)
    @pytest.mark.parametrize("merge_strategy", ["mean", "median", "sum", "max", "min"])
    def test_trainer(
        self, model_config_path: str, merge_strategy: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        base_model = instantiate(model_conf.model)
        tta_model = TTAClassification(base_model, merge_strategy=merge_strategy)
        datamodule = ToyImageClassificationDatamodule()

        trainer = Trainer(default_root_dir=str(tmp_path))

        trainer.test(tta_model, datamodule)
