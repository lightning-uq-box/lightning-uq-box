# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Image Regression Tasks."""

import glob
from pathlib import Path
from typing import Any, Dict

import pytest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyImageRegressionDatamodule
from lightning_uq_box.uq_methods import DeepEnsembleRegression

model_config_paths = [
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/mean_variance_estimation.yaml",
    "tests/configs/image_regression/qr_model.yaml",
    "tests/configs/image_regression/conformal_qr.yaml",
    "tests/configs/image_regression/der.yaml",
    "tests/configs/image_regression/bnn_vi_elbo.yaml",
    "tests/configs/image_regression/bnn_vi.yaml",
    "tests/configs/image_regression/bnn_vi_lv_last.yaml",
    "tests/configs/image_regression/swag.yaml",
    "tests/configs/image_regression/sgld_mse.yaml",
    "tests/configs/image_regression/dkl.yaml",
    "tests/configs/image_regression/due.yaml",
]

data_config_paths = ["tests/configs/image_regression/toy_image_regression.yaml"]


class TestImageRegressionTask:
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
            max_epochs=2, log_every_n_steps=1, default_root_dir=str(tmp_path)
        )
        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)


ensemble_model_config_paths = [
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/mean_variance_estimation.yaml",
]


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

    def test_deep_ensemble(self, ensemble_members_dict: Dict[str, Any]) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleRegression(
            len(ensemble_members_dict), ensemble_members_dict
        )
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer()
        trainer.test(ensemble_model, datamodule=datamodule)
