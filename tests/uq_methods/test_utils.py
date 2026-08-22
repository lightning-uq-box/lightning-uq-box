# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for UQ-Methods."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from conftest import minimal_cli_overrides

from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.uq_methods.utils import (
    checkpoint_loader,
    save_classification_predictions,
)


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
        ] + minimal_cli_overrides(
            {"accelerator": "cpu", "devices": "auto"},
            tmp_path,
            max_epochs=2,
            checkpoints=True,
        )

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)

        return cli

    def test_checkpoint_load_lightning_module(self, exp_run):
        # Get the path of the saved checkpoint
        ckpt_path = exp_run.trainer.checkpoint_callback.best_model_path
        assert ckpt_path

        model = checkpoint_loader(exp_run.model, ckpt_path=ckpt_path)

        for param_tensor in model.state_dict():
            assert torch.allclose(
                model.state_dict()[param_tensor],
                torch.load(ckpt_path, weights_only=True)["state_dict"][param_tensor],
            )

    def test_checkpoint_load_model(self, exp_run):
        # Get the path of the saved checkpoint
        ckpt_path = exp_run.trainer.checkpoint_callback.best_model_path
        assert ckpt_path

        model = checkpoint_loader(exp_run.model, ckpt_path=ckpt_path, return_model=True)

        for param_tensor in model.state_dict():
            assert torch.allclose(
                model.state_dict()[param_tensor],
                # need to prepred model for the lightning module state dict
                torch.load(ckpt_path, weights_only=True)["state_dict"][
                    "model." + param_tensor
                ],
            )


def test_save_classification_predictions(tmp_path: Path) -> None:
    """Each class_prob_i column should hold the probability of class i."""
    class_probs = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    path = str(tmp_path / "preds.csv")

    save_classification_predictions(
        {"pred": class_probs.clone(), "target": torch.tensor([2, 0])}, path
    )

    df = pd.read_csv(path)
    assert df["pred"].tolist() == [2, 0]
    for i in range(class_probs.shape[1]):
        assert df[f"class_prob_{i}"].tolist() == pytest.approx(
            class_probs[:, i].tolist()
        )
