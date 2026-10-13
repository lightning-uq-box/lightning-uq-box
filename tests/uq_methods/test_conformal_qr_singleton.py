# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Regression coverage for singleton CQR calibration sets."""

from pathlib import Path

import pytest
import torch
from lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lightning_uq_box.uq_methods.conformal_qr import ConformalQR, compute_q_hat_with_cqr


class SingletonCalibration(Dataset[dict[str, Tensor]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index != 0:
            raise IndexError(index)
        return {"input": torch.tensor([0.0]), "target": torch.tensor([3.0])}


@pytest.mark.parametrize("target, expected", [(1.0, -1.0), (2.0, 0.0), (3.0, 1.0)])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("alpha", [0.5, 0.75])
def test_singleton_cqr_score(
    target: float, expected: float, column: bool, alpha: float
) -> None:
    labels = torch.tensor([target], dtype=torch.float64)
    if column:
        labels = labels[:, None]
    result = compute_q_hat_with_cqr(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        labels,
        alpha,
    )
    assert isinstance(result, Tensor)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("column", [False, True])
def test_multiple_cqr_scores_unchanged(column: bool) -> None:
    labels = torch.tensor([3.0, 4.0, 1.0, 2.0])
    if column:
        labels = labels[:, None]
    result = compute_q_hat_with_cqr(torch.zeros(4), torch.full((4,), 2.0), labels, 0.5)
    assert isinstance(result, Tensor)
    torch.testing.assert_close(result, torch.tensor(2.0))


def test_trainer_calibrates_one_example(tmp_path: Path) -> None:
    predictor = nn.Linear(1, 3)
    with torch.no_grad():
        predictor.weight.zero_()
        assert predictor.bias is not None
        predictor.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    model = ConformalQR(predictor, alpha=0.5)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        model, train_dataloaders=DataLoader(SingletonCalibration(), batch_size=1)
    )
    assert model.post_hoc_fitted
    prediction = model.predict_step(torch.tensor([[0.0]]))
    torch.testing.assert_close(prediction["lower_quant"], torch.tensor([-1.0]))
    torch.testing.assert_close(prediction["upper_quant"], torch.tensor([3.0]))
