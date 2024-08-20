# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Deep Ensemble."""

import glob
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyImageRegressionDatamodule
from lightning_uq_box.uq_methods import DeepEnsembleRegression

data_config_paths = ["tests/configs/image_regression/toy_image_regression.yaml"]

ensemble_model_config_paths = [
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/mean_variance_estimation.yaml",
    "tests/configs/image_regression/mc_dropout_mse.yaml",
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
        for i in range(3):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.uq_method)
            datamodule = instantiate(data_conf.data)
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=2,
                log_every_n_steps=1,
                default_root_dir=str(tmp_path),
            )
            trainer.fit(model, datamodule)

            if "mc_dropout" in model_config_path:
                with pytest.raises(
                    UserWarning, match="No dropout layers found in model"
                ):
                    trainer.test(ckpt_path="best", datamodule=datamodule)
            else:
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
        ensemble_model = DeepEnsembleRegression(ensemble_members_dict)
        datamodule = ToyImageRegressionDatamodule()
        batch = next(iter(datamodule.test_dataloader()))

        if "mc_dropout" in ensemble_members_dict[0]["ckpt_path"]:
            with pytest.raises(UserWarning, match="No dropout layers found in model"):
                pred = ensemble_model.predict_step(batch["input"])
        else:
            pred = ensemble_model.predict_step(batch["input"])

        # check that prediction under ensemble are the same as the individual ones
        for i in range(len(ensemble_members_dict)):
            with torch.no_grad():
                assert torch.allclose(
                    pred["samples"][..., i],
                    ensemble_members_dict[i]["base_model"](batch["input"]),
                )
