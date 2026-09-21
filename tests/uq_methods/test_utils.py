# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Utilities for UQ-Methods."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import minimal_cli_overrides

import lightning_uq_box.uq_methods.utils as uq_utils
from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.uq_methods.utils import (
    checkpoint_loader,
    default_classification_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    process_classification_prediction,
    process_segmentation_prediction,
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


# "binary" is left out: EmpiricalCoverage needs a [batch_size, num_classes]
# prediction tensor while BinaryAccuracy needs one shaped like the target, so the
# binary metric set cannot be updated with a single prediction tensor. That is a
# pre-existing incompatibility, unrelated to multilabel support.
@pytest.mark.parametrize("task", ["multiclass", "multilabel"])
def test_default_classification_metrics(task: str) -> None:
    """The default metrics construct and update for every supported task."""
    num_classes = 4
    metrics = default_classification_metrics("test", task, num_classes)

    preds = torch.rand(8, num_classes)
    if task == "multiclass":
        target = torch.randint(0, num_classes, (8,))
    else:
        target = torch.randint(0, 2, (8, num_classes))

    assert metrics(preds, target)


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_default_segmentation_metrics(task: str) -> None:
    """The default metrics construct and update for every supported task."""
    num_classes = 1 if task == "binary" else 4
    metrics = default_segmentation_metrics("test", task, num_classes)

    preds = (
        torch.rand(2, 4, 4) if task == "binary" else torch.rand(2, num_classes, 4, 4)
    )
    if task == "multiclass":
        target = torch.randint(0, num_classes, (2, 4, 4))
    elif task == "binary":
        target = torch.randint(0, 2, (2, 4, 4))
    else:
        target = torch.randint(0, 2, (2, num_classes, 4, 4))

    assert metrics(preds, target)


def test_process_classification_prediction_multilabel() -> None:
    """Multilabel probabilities are per label and need not sum to one."""
    logits = torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]).unsqueeze(-1)

    out = process_classification_prediction(logits, task="multilabel")

    assert out["pred"].shape == (2, 3)
    assert torch.allclose(out["pred"], torch.sigmoid(logits[..., 0]))
    assert out["pred_uct"].shape == (2,)


def test_save_classification_predictions_multilabel(tmp_path: Path) -> None:
    """Multilabel predictions are saved as one thresholded column per label."""
    class_probs = torch.tensor([[0.1, 0.8, 0.7], [0.6, 0.3, 0.1]])
    target = torch.tensor([[0, 1, 1], [1, 0, 0]])
    path = str(tmp_path / "preds.csv")

    save_classification_predictions(
        {"pred": class_probs.clone(), "target": target}, path, task="multilabel"
    )

    df = pd.read_csv(path)
    assert "pred" not in df.columns
    for i in range(class_probs.shape[1]):
        assert df[f"pred_{i}"].tolist() == (class_probs[:, i] > 0.5).int().tolist()
        assert df[f"target_{i}"].tolist() == target[:, i].tolist()
        assert df[f"class_prob_{i}"].tolist() == pytest.approx(
            class_probs[:, i].tolist()
        )


def test_binary_prediction_helpers_preserve_single_item_batches() -> None:
    """One-logit vector and dense conversions never squeeze the batch axis."""
    classification_logits = torch.tensor([[[1.0, -1.0]]])
    classification = process_classification_prediction(
        classification_logits, task="binary", binary_encoding="one_logit"
    )
    assert classification["pred"].shape == (1, 1)
    assert classification["pred_uct"].shape == (1,)

    segmentation_logits = torch.zeros(1, 1, 2, 2, 2)
    segmentation = process_segmentation_prediction(
        segmentation_logits, task="binary", binary_encoding="one_logit"
    )
    assert segmentation["pred"].shape == (1, 1, 2, 2)
    assert segmentation["pred_uct"].shape == (1, 2, 2)


def test_canonical_metric_and_writer_helpers_cover_edge_shapes(tmp_path: Path) -> None:
    """Canonical helpers support B=1 without restoring legacy squeeze behavior."""
    assert "R2" in default_regression_metrics("test")
    assert "R2" not in default_regression_metrics("test", include_r2=False)
    binary = default_classification_metrics("test", "binary", 1)
    assert binary(torch.tensor([0.8]), torch.tensor([1]))
    assert uq_utils._writer_array(torch.ones(1, 1)).shape == (1,)
    assert uq_utils._writer_array(torch.tensor(1.0)).shape == (1,)
    assert np.array_equal(uq_utils._writer_array([1, 2]), np.array([1, 2]))

    outputs = {
        "pred": torch.tensor([0.8]),
        "target": torch.tensor([1]),
        "samples": torch.tensor([[0.8, 0.7]]),
        "logits": torch.tensor([[1.0]]),
        "pred_set": [torch.tensor([True, False])],
    }
    path = tmp_path / "nested" / "preds.csv"
    save_classification_predictions(
        outputs, str(path), task="binary", binary_encoding="one_logit"
    )
    assert set(outputs) == {"pred", "target", "samples", "logits", "pred_set"}
    assert pd.read_csv(path)["pred"].tolist() == [1]


def test_distributed_paths_are_rank_sharded_with_a_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Distributed writers never share a filename and rank zero publishes discovery."""
    monkeypatch.setattr(uq_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(uq_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(uq_utils.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(uq_utils.torch.distributed, "get_rank", lambda: 0)
    path = tmp_path / "preds.csv"
    assert uq_utils._distributed_path(str(path)) == str(tmp_path / "preds.rank-0.csv")
    manifest = json.loads((tmp_path / "preds.manifest.json").read_text())
    assert manifest == {
        "version": 1,
        "world_size": 2,
        "shards": ["preds.rank-0.csv", "preds.rank-1.csv"],
    }
    monkeypatch.setattr(uq_utils.torch.distributed, "get_rank", lambda: 1)
    assert uq_utils._distributed_path(str(path)) == str(tmp_path / "preds.rank-1.csv")


def test_distributed_path_keeps_single_process_filename(
    monkeypatch, tmp_path: Path
) -> None:
    """Unavailable and single-rank distributed backends preserve legacy paths."""
    path = str(tmp_path / "preds.csv")
    monkeypatch.setattr(uq_utils.torch.distributed, "is_available", lambda: False)
    assert uq_utils._distributed_path(path) == path
    monkeypatch.setattr(uq_utils.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(uq_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(uq_utils.torch.distributed, "get_world_size", lambda: 1)
    assert uq_utils._distributed_path(path) == path
