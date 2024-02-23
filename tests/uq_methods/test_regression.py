# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Regression Tasks."""

import glob
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from lightning import Trainer
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.uq_methods import DeepEnsembleRegression

model_config_paths = [
    "tests/configs/regression/mc_dropout_mse.yaml",
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/mean_variance_estimation.yaml",
    "tests/configs/regression/qr_model.yaml",
    "tests/configs/regression/conformal_qr.yaml",
    "tests/configs/regression/conformal_qr_with_module.yaml",
    "tests/configs/regression/der.yaml",
    "tests/configs/regression/bnn_vi_elbo.yaml",
    "tests/configs/regression/bnn_vi.yaml",
    "tests/configs/regression/bnn_vi_batched.yaml",
    "tests/configs/regression/bnn_vi_lv_first_batched.yaml",
    "tests/configs/regression/bnn_vi_lv_first.yaml",
    "tests/configs/regression/bnn_vi_lv_last.yaml",
    "tests/configs/regression/swag.yaml",
    "tests/configs/regression/sgld_mse.yaml",
    "tests/configs/regression/sgld_nll.yaml",
    "tests/configs/regression/dkl.yaml",
    "tests/configs/regression/due.yaml",
    "tests/configs/regression/cards.yaml",
    "tests/configs/regression/sngp.yaml",
]

data_config_paths = ["tests/configs/regression/toy_regression.yaml"]


class TestRegressionTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self, model_config_path: str, data_config_path: str, tmp_path: Path
    ) -> None:
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "2",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.logger",
            "CSVLogger",
            "--trainer.logger.save_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        if "laplace" not in model_config_path:
            cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

        # assert predictions are saved
        assert os.path.exists(
            os.path.join(cli.trainer.default_root_dir, cli.model.pred_file_name)
        )


ensemble_model_config_paths = [
    "tests/configs/regression/mc_dropout_mse.yaml",
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/mean_variance_estimation.yaml",
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
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(5):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            args = [
                "--config",
                model_config_path,
                "--config",
                data_config_path,
                "--trainer.accelerator",
                "cpu",
                "--trainer.max_epochs",
                "1",
                "--trainer.log_every_n_steps",
                "1",
                "--trainer.default_root_dir",
                str(tmp_path),
            ]

            cli = get_uq_box_cli(args)
            cli.trainer.fit(cli.model, cli.datamodule)

            # Find the .ckpt file in the lightning_logs directory
            ckpt_file = glob.glob(
                f"{str(tmp_path)}/lightning_logs/version_*/checkpoints/*.ckpt"
            )[0]
            ckpt_paths.append({"base_model": cli.model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self, ensemble_members_dict: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleRegression(
            len(ensemble_members_dict), ensemble_members_dict
        )

        datamodule = ToyHeteroscedasticDatamodule()

        trainer = Trainer(default_root_dir=str(tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)

        assert os.path.exists(
            os.path.join(trainer.default_root_dir, ensemble_model.pred_file_name)
        )
