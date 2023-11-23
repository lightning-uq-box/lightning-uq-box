"""Test Regression Tasks."""

import glob
import os
from itertools import product
from pathlib import Path

import pytest

from lightning_uq_box.main import get_uq_box_cli

model_config_paths = [
    "tests/configs/regression/mc_dropout_mse.yaml",
    "tests/configs/regression/mc_dropout_nll.yaml",
    "tests/configs/regression/mean_variance_estimation.yaml",
    "tests/configs/regression/qr_model.yaml",
    "tests/configs/regression/conformal_qr.yaml",
    "tests/configs/regression/der.yaml",
    "tests/configs/regression/bnn_vi_elbo.yaml",
    "tests/configs/regression/bnn_vi.yaml",
    "tests/configs/regression/bnn_vi_lv_first.yaml",
    "tests/configs/regression/bnn_vi_lv_last.yaml",
    "tests/configs/regression/card_linear.yaml",
    "tests/configs/regression/swag.yaml",
    "tests/configs/regression/sgld_mse.yaml",
    "tests/configs/regression/sgld_nll.yaml",
    "tests/configs/regression/dkl.yaml",
    "tests/configs/regression/due.yaml",
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
            "1",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
