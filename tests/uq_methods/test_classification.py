"""Test Classification Tasks."""

import glob
import os
from pathlib import Path

import pytest

from lightning_uq_box.main import get_uq_box_cli

model_config_paths = [
    "tests/configs/classification/mc_dropout.yaml",
    "tests/configs/classification/bnn_vi_elbo.yaml",
    "tests/configs/classification/swag.yaml",
    "tests/configs/classification/sgld.yaml",
    "tests/configs/classification/dkl.yaml",
    # "tests/configs/classification/due.yaml"
]

data_config_paths = ["tests/configs/classification/toy_classification.yaml"]


class TestClassificationTask:
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
