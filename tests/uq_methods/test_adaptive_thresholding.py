# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import h5py
import pytest
import torch
from lightning import Trainer
from torch import Tensor, nn

from lightning_uq_box.datamodules import ToySegmentationDataModule
from lightning_uq_box.models import ThresholdPredictor
from lightning_uq_box.uq_methods import AdaptiveThresholding
from lightning_uq_box.uq_methods.metrics import CoverageGap, PerImageCoverage
from lightning_uq_box.uq_methods.segmentation_conformal_base import (
    SegmentationPosthocBase,
)
from lightning_uq_box.uq_methods.segmentation_conformal_utils import (
    compute_oracle_threshold,
    find_calibration_shift,
    per_image_coverage,
)


class TinyThreshold(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, image: Tensor, phat: Tensor) -> Tensor:
        return self.value.sigmoid().expand(image.shape[0])


def test_oracle_and_ties() -> None:
    phat = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 1, 2, 2)
    labels = torch.ones_like(phat)
    tau = compute_oracle_threshold(phat, labels, alpha=0.5)
    assert tau.item() == pytest.approx(0.3, abs=1e-6)
    assert per_image_coverage(phat >= tau[:, None, None, None], labels).item() == 0.5
    tied = torch.full_like(phat, 0.2)
    assert (tied >= compute_oracle_threshold(tied, labels)[:, None, None, None]).all()
    assert compute_oracle_threshold(phat, torch.zeros_like(labels)).item() == 1


def test_calibration() -> None:
    phat = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, 1, 2, 2).expand(10, -1, -1, -1)
    labels = torch.ones_like(phat)
    tau = torch.full((10,), 0.8)
    shift = find_calibration_shift(tau, phat, labels, target_coverage=0.5)
    assert shift == pytest.approx(0.6, abs=1e-4)
    coverage = per_image_coverage(
        phat >= (tau - shift)[:, None, None, None], labels
    ).mean()
    assert coverage >= 0.55
    assert find_calibration_shift(tau[:1], phat[:1], labels[:1], 0.9) == 1
    with pytest.raises(ValueError):
        find_calibration_shift(tau, phat, labels, 0.9, search_range=(-0.1, 0.1))


def test_gap_is_mean_absolute_gap() -> None:
    pred = torch.tensor([0, 1]).view(2, 1, 1, 1).bool()
    labels = torch.ones_like(pred)
    gap, coverage = CoverageGap(0.5), PerImageCoverage()
    cast(Callable[..., None], gap.update)(pred, labels)
    cast(Callable[..., None], coverage.update)(pred, labels)
    assert cast(Callable[[], Tensor], gap.compute)().item() == 0.5
    assert cast(Callable[[], Tensor], coverage.compute)().item() == 0.5
    gap.reset()
    cast(Callable[..., None], gap.update)(
        torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    )
    assert cast(Callable[[], Tensor], gap.compute)().item() == 0.5


def test_threshold_predictor() -> None:
    model = ThresholdPredictor(pretrained=False).eval()
    weight = model.resnet.conv1.weight
    assert weight.shape[1] == 4
    torch.testing.assert_close(weight[:, 3:4], weight[:, :3].mean(1, keepdim=True))
    with torch.no_grad():
        tau = model(torch.randn(2, 3, 32, 32), torch.rand(2, 16, 16))
    assert tau.shape == (2,)
    assert ((tau >= 0) & (tau <= 1)).all()


def run_two_stage(method: SegmentationPosthocBase, tmp_path: Path) -> None:
    dm = ToySegmentationDataModule(
        num_images=12, image_size=16, batch_size=4, num_classes=2
    )
    base_before = {k: v.clone() for k, v in method.model.state_dict().items()}
    threshold_model = cast(TinyThreshold, method.threshold_model)
    before = threshold_model.value.detach().clone()
    with pytest.raises(RuntimeError, match="post hoc fitted"):
        method.predict_step(torch.rand(2, 3, 16, 16))
    settings: dict[str, Any] = {
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "default_root_dir": str(tmp_path),
        "accelerator": "cpu",
        "num_sanity_val_steps": 0,
    }
    first = Trainer(**settings)
    first.fit(method, train_dataloaders=dm.train_dataloader())
    assert method._network_trained and not method.post_hoc_fitted
    assert threshold_model.value.detach() != before
    for k, v in method.model.state_dict().items():
        torch.testing.assert_close(v, base_before[k])
    assert all(p.grad is None for p in method.model.parameters())
    with pytest.raises(RuntimeError):
        method(torch.rand(2, 3, 16, 16))
    # A fresh Trainer is essential: the previous Trainer has exhausted max_epochs.
    second = Trainer(**settings)
    second.fit(method, train_dataloaders=dm.calib_dataloader())
    assert method.post_hoc_fitted
    second.test(method, dataloaders=dm.test_dataloader())
    with h5py.File(next((tmp_path / "preds").glob("*.hdf5"))) as f:
        assert f["pred"].shape == (1, 16, 16)
        assert f["phat"].shape == f["target"].shape
        assert "tau" in f.attrs
    checkpoint = tmp_path / "calibrated.ckpt"
    second.save_checkpoint(checkpoint)
    with patch(
        f"{type(method).__module__}.ThresholdPredictor", return_value=TinyThreshold()
    ):
        restored = type(method).load_from_checkpoint(
            checkpoint,
            model=nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
            pretrained_threshold_net=False,
        )
    assert restored.post_hoc_fitted and restored._network_trained
    torch.testing.assert_close(restored.t_prime, method.t_prime)
    X = torch.rand(2, 3, 16, 16)
    restored.eval()
    torch.testing.assert_close(restored(X)["pred"], method(X)["pred"])


def test_two_stage(tmp_path: Path) -> None:
    with patch(
        "lightning_uq_box.uq_methods.adaptive_thresholding.ThresholdPredictor",
        return_value=TinyThreshold(),
    ):
        method = AdaptiveThresholding(
            nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
            pretrained_threshold_net=False,
            save_preds=True,
        )
    run_two_stage(method, tmp_path)


def test_config_instantiation() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(
        "tests/configs/image_segmentation/adaptive_thresholding.yaml"
    )
    method = instantiate(config.uq_method)
    dm = instantiate(config.data)
    batch = next(iter(dm.train_dataloader()))
    loss = method._training_step_network(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in method.threshold_model.parameters())
    assert all(p.grad is None for p in method.model.parameters())
