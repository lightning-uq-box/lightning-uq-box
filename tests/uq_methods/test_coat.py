# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lightning_uq_box.uq_methods import COAT
from lightning_uq_box.uq_methods.loss_functions import SoftMiscoverageLoss
from tests.uq_methods.test_adaptive_thresholding import TinyThreshold, run_two_stage


def test_loop_equivalence() -> None:
    torch.manual_seed(10)
    phat = torch.rand(4, 1, 5, 5)
    labels = torch.randint(2, phat.shape).float()
    labels[0].zero_()
    tau = torch.rand(4, requires_grad=True)
    loss_fn = SoftMiscoverageLoss()
    expected = torch.stack(
        [
            (
                (
                    (torch.sigmoid((phat[i] - tau[i]) / 0.05) * labels[i]).sum()
                    / (labels[i].sum() + 1e-6)
                    - 0.9
                )
                ** 2
            )
            for i in range(4)
        ]
    ).mean()
    actual = loss_fn(phat, labels, tau)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, tau, retain_graph=True)[0],
        torch.autograd.grad(expected, tau)[0],
    )
    torch.testing.assert_close(actual, loss_fn(phat[:, 0], labels[:, 0], tau))


def test_gradient_direction() -> None:
    phat = torch.full((2, 1, 3, 3), 0.5)
    labels = torch.ones_like(phat)
    tau = torch.full((2,), 0.7, requires_grad=True)
    loss_fn = SoftMiscoverageLoss()
    loss = loss_fn(phat, labels, tau)
    loss.backward()
    assert tau.grad is not None and torch.isfinite(tau.grad).all()
    assert (tau.grad > 0).all()
    updated = tau.detach() - 0.01 * tau.grad
    assert (updated < tau.detach()).all()
    assert loss_fn(phat, labels, updated) < loss


def test_hard_limit() -> None:
    phat = torch.tensor([0.1, 0.3, 0.7, 0.9]).view(1, 1, 2, 2)
    labels = torch.ones_like(phat)
    tau = torch.tensor([0.5])
    expected = ((phat >= 0.5).float().mean() - 0.9) ** 2
    torch.testing.assert_close(
        SoftMiscoverageLoss(temperature=1e-5)(phat, labels, tau), expected
    )


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        SoftMiscoverageLoss(temperature)


def test_empty_foreground() -> None:
    tau = torch.tensor([0.5], requires_grad=True)
    loss = SoftMiscoverageLoss()(torch.rand(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), tau)
    loss.backward()
    assert loss.item() == pytest.approx(0.81)
    torch.testing.assert_close(tau.grad, torch.zeros(1))


def test_two_stage_coat(tmp_path: Path) -> None:
    with patch(
        "lightning_uq_box.uq_methods.coat.ThresholdPredictor",
        return_value=TinyThreshold(),
    ):
        method = COAT(
            nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
            alpha=0.2,
            lr=0.003,
            temperature=0.07,
            pretrained_threshold_net=False,
            save_preds=True,
        )
    run_two_stage(method, tmp_path)
    with patch(
        "lightning_uq_box.uq_methods.coat.ThresholdPredictor",
        return_value=TinyThreshold(),
    ):
        restored = COAT.load_from_checkpoint(
            tmp_path / "calibrated.ckpt",
            model=nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
        )
    assert restored.soft_miscoverage_loss.temperature == 0.07
    assert restored.soft_miscoverage_loss.target_coverage == 0.8


def test_config_instantiation() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load("tests/configs/image_segmentation/coat.yaml")
    method = instantiate(config.uq_method)
    dm = instantiate(config.data)
    loss = method._training_step_network(next(iter(dm.train_dataloader())))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in method.threshold_model.parameters())
    assert all(p.grad is None for p in method.model.parameters())
