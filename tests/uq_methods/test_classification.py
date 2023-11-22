"""Test Classification Tasks."""

import glob
import os
from pathlib import Path

import pytest

from lightning_uq_box.main import main


class TestRegressionTask:
    # @pytest.mark.parametrize(
    #     "config_path", glob.glob(os.path.join("tests", "configs", "classification", "*.yaml"))
    # )
    @pytest.mark.parametrize(
        "config_path",
        [
            "/home/nils/projects/lightning-uq-box/tests/configs/classification/bnn_vi_elbo.yaml",
            # "/home/nils/projects/lightning-uq-box/tests/configs/classification/dkl.yaml",
            # "/home/nils/projects/lightning-uq-box/tests/configs/classification/due.yaml",
            # "/home/nils/projects/lightning-uq-box/tests/configs/classification/laplace.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/classification/mc_dropout.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/classification/sgld.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/classification/swag.yaml",
        ],
    )
    def test_trainer(self, config_path: str, tmp_path: Path) -> None:
        args = [
            "--config",
            config_path,
            "--config",
            "/home/nils/projects/lightning-uq-box/tests/configs/classification/toy_classification.yaml",
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "1",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
        ]

        # TODO should probably do this: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#instantiation-only-mode
        # in order to test both training and test
        main(["fit"] + args)
