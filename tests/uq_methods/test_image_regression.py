# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Image Regression Tasks."""

import glob
import os
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyImageRegressionDatamodule
from lightning_uq_box.uq_methods import DeepEnsembleRegression, TTARegression

model_config_paths = [
    "tests/configs/image_regression/mean_variance_estimation.yaml",
    "tests/configs/image_regression/qr_model.yaml",
    "tests/configs/image_regression/der.yaml",
    "tests/configs/image_regression/bnn_vi_elbo.yaml",
    "tests/configs/image_regression/bnn_vi.yaml",
    "tests/configs/image_regression/bnn_vi_lv_last.yaml",
    "tests/configs/image_regression/swag.yaml",
    "tests/configs/image_regression/sgld_mse.yaml",
    "tests/configs/image_regression/dkl.yaml",
    "tests/configs/image_regression/due.yaml",
    "tests/configs/image_regression/laplace_glm.yaml",
    "tests/configs/image_regression/laplace_nn.yaml",
    "tests/configs/image_regression/cards.yaml",
    "tests/configs/image_regression/sngp.yaml",
    "tests/configs/image_regression/vbll_replace.yaml",
    "tests/configs/image_regression/vbll_attach.yaml",
    "tests/configs/image_regression/masked_ensemble.yaml",
    "tests/configs/image_regression/zigzag.yaml",
    "tests/configs/image_regression/mixture_density.yaml",
    "tests/configs/image_regression/density_layer.yaml",
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

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            accelerator="cpu",
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

        assert os.path.exists(
            os.path.join(trainer.default_root_dir, model.pred_file_name)
        )

        df = pd.read_csv(os.path.join(trainer.default_root_dir, model.pred_file_name))
        if "qr_model" not in model_config_path:
            assert (df["pred_uct"] > 0).all()


mc_dropout_config_paths = [
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/mc_dropout_mse.yaml",
]


class TestMCDropout:
    @pytest.mark.parametrize("model_config_path", mc_dropout_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            log_every_n_steps=1,
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )
        with pytest.raises(UserWarning, match="No dropout layers found in model"):
            trainer.fit(model, datamodule)
            trainer.test(ckpt_path="best", datamodule=datamodule)


posthoc_config_paths = ["tests/configs/image_regression/conformal_qr.yaml"]


class TestPosthoc:
    @pytest.mark.parametrize("model_config_path", posthoc_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    @pytest.mark.parametrize("calibration", [True, False])
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        calibration: bool,
        tmp_path: Path,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            default_root_dir=str(tmp_path),
            accelerator="cpu",
            max_epochs=1,
            log_every_n_steps=1,
        )
        if calibration:
            trainer.fit(model, train_dataloaders=datamodule.calib_dataloader())
            trainer.test(model, datamodule=datamodule)
        else:
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Model has not been post hoc fitted, please call trainer.fit(model, train_dataloaders=dm.calib_dataloader()) first."
                ),
            ):
                X = torch.rand(1, 3, 64, 64)
                model(X)


ensemble_model_config_paths = [
    "tests/configs/image_regression/mean_variance_estimation.yaml"
]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(
        self, request, tmp_path_factory: TempPathFactory
    ) -> list[dict[str, Any]]:
        model_config_path, data_config_path = request.param
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(5):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.uq_method)
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
        self, ensemble_members_dict: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleRegression(ensemble_members_dict)
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(accelerator="cpu", default_root_dir=str(tmp_path))
        if "mc_dropout" in ensemble_members_dict[0]["ckpt_path"]:
            with pytest.raises(UserWarning, match="No dropout layers found in model"):
                trainer.test(ensemble_model, datamodule)
        else:
            trainer.test(ensemble_model, datamodule=datamodule)

        # check that predictions are saved
        assert os.path.exists(
            os.path.join(trainer.default_root_dir, ensemble_model.pred_file_name)
        )


tta_model_paths = [
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/qr_model.yaml",
    "tests/configs/image_regression/tta_augmentation.yaml",
]


class TestTTAModel:
    @pytest.mark.parametrize("model_config_path", tta_model_paths)
    @pytest.mark.parametrize("merge_strategy", ["mean", "median", "sum", "max", "min"])
    def test_trainer(
        self,
        model_config_path: str,
        merge_strategy: Literal["mean", "median", "sum", "max", "min"],
        tmp_path: Path,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        base_model = instantiate(model_conf.uq_method)
        tta_model = TTARegression(base_model, merge_strategy=merge_strategy)
        datamodule = ToyImageRegressionDatamodule()

        trainer = Trainer(accelerator="cpu", default_root_dir=str(tmp_path))

        if "mc_dropout" in model_config_path:
            with pytest.raises(UserWarning, match="No dropout layers found in model"):
                trainer.test(tta_model, datamodule)
        else:
            trainer.test(tta_model, datamodule)


frozen_config_paths = [
    "tests/configs/image_regression/mean_variance_estimation.yaml",
    "tests/configs/image_regression/mc_dropout_nll.yaml",
    "tests/configs/image_regression/bnn_vi_elbo.yaml",
    "tests/configs/image_regression/bnn_vi.yaml",
    "tests/configs/image_regression/due.yaml",
    "tests/configs/image_regression/sngp.yaml",
    "tests/configs/image_regression/der.yaml",
    "tests/configs/image_regression/mixture_density.yaml",
]


class TestFrozenBackbone:
    @pytest.mark.parametrize("model_name", ["resnet18", "vit_small_patch8_224"])
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(self, model_config_path: str, model_name: str) -> None:
        model_conf = OmegaConf.load(model_config_path)

        try:
            model_conf.uq_method.model.model_name = model_name
            model = instantiate(model_conf.uq_method, freeze_backbone=True)
            assert not all([param.requires_grad for param in model.model.parameters()])
            assert all(
                [
                    param.requires_grad
                    for param in model.model.get_classifier().parameters()
                ]
            )
        except AttributeError:
            model_conf.uq_method.feature_extractor.model_name = model_name
            model_conf.uq_method.input_size = 224
            model = instantiate(model_conf.uq_method, freeze_backbone=True)
            # check that entire feature extractor is frozen
            assert not all(
                [param.requires_grad for param in model.feature_extractor.parameters()]
            )
