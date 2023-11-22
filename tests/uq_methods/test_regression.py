"""Test Regression Tasks."""

import glob
import os
from pathlib import Path

import pytest

from lightning_uq_box.main import get_uq_box_cli


class TestRegressionTask:
    # @pytest.mark.parametrize(
    #     "config_path", glob.glob(os.path.join("tests", "configs", "regression", "*.yaml"))
    # )
    @pytest.mark.parametrize(
        "config_path",
        [
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/mc_dropout_mse.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/mc_dropout_nll.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/mean_variance_estimation.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/qr_model.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/der.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/bnn_vi_elbo.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/bnn_vi.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/bnn_vi_lv_first.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/bnn_vi_lv_last.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/card_linear.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/swag.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/sgld_mse.yaml",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/sgld_nll.yaml",
            # # "/home/nils/projects/lightning-uq-box/tests/configs/regression/laplace.yaml"
            # "/home/nils/projects/lightning-uq-box/tests/configs/regression/dkl.yaml"
            # "/home/nils/projects/lightning-uq-box/tests/configs/regression/due.yaml"
        ],
    )
    def test_trainer(self, config_path: str, tmp_path: Path) -> None:
        args = [
            "--config",
            config_path,
            "--config",
            "/home/nils/projects/lightning-uq-box/tests/configs/regression/toy_regression.yaml",
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
