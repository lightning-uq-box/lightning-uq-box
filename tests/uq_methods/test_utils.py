# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for UQ-Methods."""

import os
from glob import glob
from pathlib import Path

import pytest
import torch

from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.uq_methods.utils import checkpoint_loader


class TestUQMethods:
    @pytest.fixture(
        params=[
            (
                "tests/configs/regression/mc_dropout_nll.yaml",
                "tests/configs/regression/toy_regression.yaml",
            )
        ]
    )
    def exp_run(self, request, tmp_path: Path):
        model_config_path, data_config_path = request.param
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
            "--trainer.callbacks+=ModelCheckpoint",
            "--trainer.callbacks.dirpath",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)

        return cli

    def test_checkpoint_load_lightning_module(self, exp_run):
        # Get the path of the saved checkpoint
        ckpt_path = glob(os.path.join(exp_run.trainer.default_root_dir, "*ckpt"))[0]

        model = checkpoint_loader(exp_run.model, ckpt_path=ckpt_path)

        for param_tensor in model.state_dict():
            assert torch.allclose(
                model.state_dict()[param_tensor],
                torch.load(ckpt_path)["state_dict"][param_tensor],
            )

    def test_checkpoint_load_model(self, exp_run):
        # Get the path of the saved checkpoint
        ckpt_path = glob(os.path.join(exp_run.trainer.default_root_dir, "*ckpt"))[0]

        model = checkpoint_loader(exp_run.model, ckpt_path=ckpt_path, return_model=True)

        for param_tensor in model.state_dict():
            assert torch.allclose(
                model.state_dict()[param_tensor],
                # need to prepred model for the lightning module state dict
                torch.load(ckpt_path)["state_dict"]["model." + param_tensor],
            )
