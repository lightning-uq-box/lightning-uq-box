# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import h5py
import lightning
import pytest
import torch
from lightning import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

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
    assert model.in_channels is None
    with torch.no_grad():
        tau = model(torch.randn(2, 3, 32, 32), torch.rand(2, 16, 16))
    assert tau.shape == (2,)
    assert ((tau >= 0) & (tau <= 1)).all()
    assert model.in_channels == 3
    weight = model.resnet.conv1.weight
    torch.testing.assert_close(weight[:, 3:4], weight[:, :3].mean(1, keepdim=True))


@pytest.mark.parametrize("in_channels", [None, 3])
def test_threshold_predictor_rgb_stem_unchanged(in_channels: int | None) -> None:
    torch.manual_seed(0)
    model = ThresholdPredictor(pretrained=False, in_channels=in_channels)
    torch.manual_seed(0)
    original = resnet50(weights=None).conv1.weight
    model(torch.randn(2, 3, 32, 32), torch.rand(2, 1, 32, 32))
    weight = model.resnet.conv1.weight
    assert torch.equal(weight[:, :3], original)
    assert torch.equal(weight[:, 3:], original.mean(dim=1, keepdim=True))


@pytest.mark.parametrize("in_channels", [None, 6])
def test_threshold_predictor_six_channels(in_channels: int | None) -> None:
    model = ThresholdPredictor(pretrained=False, in_channels=in_channels).eval()
    with torch.no_grad():
        tau = model(torch.randn(2, 6, 32, 32), torch.rand(2, 1, 32, 32))
    assert tau.shape == (2,)
    assert model.in_channels == 6
    weight = model.resnet.conv1.weight
    assert weight.shape[1] == 7
    torch.testing.assert_close(weight[:, :3], weight[:, 3:6])
    torch.testing.assert_close(
        weight[:, 6:7], 2 * weight[:, :3].mean(dim=1, keepdim=True)
    )
    with pytest.raises(ValueError, match=r"\[B,6,H,W\]"):
        model(torch.randn(2, 3, 32, 32), torch.rand(2, 1, 32, 32))


def test_threshold_predictor_invalid_channels() -> None:
    with pytest.raises(ValueError, match="positive"):
        ThresholdPredictor(pretrained=False, in_channels=0)


def test_threshold_predictor_optimizer_before_first_batch() -> None:
    # Lightning builds the optimizer before any batch; the inferred stem must train.
    model = ThresholdPredictor(pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model(torch.randn(2, 6, 32, 32), torch.rand(2, 1, 32, 32)).sum().backward()
    before = model.resnet.conv1.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.resnet.conv1.weight, before)


def test_threshold_predictor_reload_before_first_batch() -> None:
    trained = ThresholdPredictor(pretrained=False).eval()
    X, phat = torch.randn(2, 6, 32, 32), torch.rand(2, 1, 32, 32)
    trained(X, phat)
    restored = ThresholdPredictor(pretrained=False).eval()
    restored.load_state_dict(trained.state_dict())
    assert restored.in_channels == 6
    with torch.no_grad():
        torch.testing.assert_close(restored(X, phat), trained(X, phat))


def save_calibrated_checkpoint(method: SegmentationPosthocBase, path: Path) -> None:
    """Write the checkpoint Trainer.save_checkpoint would write after calibration."""
    method._network_trained = True
    method.post_hoc_fitted = True
    method.t_prime.fill_(0.125)
    checkpoint: dict[str, Any] = {
        "state_dict": method.state_dict(),
        "hyper_parameters": dict(method.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    method.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)


def assert_reload_matches(method: SegmentationPosthocBase, path: Path) -> None:
    base = method.model
    save_calibrated_checkpoint(method, path)
    restored = type(method).load_from_checkpoint(
        path, model=base, threshold_model=ThresholdPredictor(pretrained=False)
    )
    assert cast(ThresholdPredictor, restored.threshold_model).in_channels == 6
    assert restored.post_hoc_fitted and restored._network_trained
    torch.testing.assert_close(restored.t_prime, method.t_prime)
    method.eval()
    restored.eval()
    X = torch.rand(2, 6, 16, 16)
    expected, actual = method(X), restored(X)
    for key in ("pred", "tau"):
        torch.testing.assert_close(actual[key], expected[key])


def test_reload_six_channels(tmp_path: Path) -> None:
    method = AdaptiveThresholding(
        nn.Sequential(nn.Conv2d(6, 1, 1), nn.BatchNorm2d(1)),
        threshold_model=ThresholdPredictor(pretrained=False),
    )
    # The threshold network sizes its stem from the first batch it sees.
    method.threshold_model(torch.rand(2, 6, 16, 16), torch.rand(2, 1, 16, 16))
    assert_reload_matches(method, tmp_path / "at.ckpt")


def test_six_channel_fit_infers_threshold_width(tmp_path: Path) -> None:
    torch.manual_seed(0)
    samples = [
        {"input": torch.rand(6, 16, 16), "target": torch.rand(16, 16) > 0.5}
        for _ in range(6)
    ]

    class SixChannels(Dataset[dict[str, Tensor]]):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict[str, Tensor]:
            return samples[index]

    data = DataLoader(SixChannels(), batch_size=2)
    method = AdaptiveThresholding(
        nn.Sequential(nn.Conv2d(6, 1, 1), nn.BatchNorm2d(1)),
        lr=1e-3,
        threshold_model=ThresholdPredictor(pretrained=False),
    )
    settings: dict[str, Any] = {
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "default_root_dir": str(tmp_path),
        "accelerator": "cpu",
    }
    Trainer(**settings).fit(method, train_dataloaders=data)
    assert cast(ThresholdPredictor, method.threshold_model).in_channels == 6
    second = Trainer(**settings)
    second.fit(method, train_dataloaders=data)
    assert method.post_hoc_fitted
    checkpoint = tmp_path / "six.ckpt"
    second.save_checkpoint(checkpoint)
    restored = AdaptiveThresholding.load_from_checkpoint(
        checkpoint,
        model=method.model,
        threshold_model=ThresholdPredictor(pretrained=False),
    )
    method.eval()
    restored.eval()
    X = torch.stack([sample["input"] for sample in samples[:2]])
    torch.testing.assert_close(restored(X)["tau"], method(X)["tau"])


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
    restored = type(method).load_from_checkpoint(
        checkpoint,
        model=nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
        threshold_model=TinyThreshold(),
    )
    assert restored.post_hoc_fitted and restored._network_trained
    assert restored.alpha == method.alpha
    assert restored.lr == method.lr
    torch.testing.assert_close(restored.t_prime, method.t_prime)
    X = torch.rand(2, 3, 16, 16)
    restored.eval()
    torch.testing.assert_close(restored(X)["pred"], method(X)["pred"])


def test_two_stage(tmp_path: Path) -> None:
    method = AdaptiveThresholding(
        nn.Sequential(nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1)),
        alpha=0.2,
        lr=0.003,
        save_preds=True,
        threshold_model=TinyThreshold(),
    )
    run_two_stage(method, tmp_path)


def test_default_threshold_model() -> None:
    # The default is the paper's network with ImageNet weights; avoid the download.
    with patch(
        "lightning_uq_box.uq_methods.adaptive_thresholding.ThresholdPredictor",
        return_value=TinyThreshold(),
    ) as default:
        method = AdaptiveThresholding(nn.Conv2d(3, 1, 1))
    default.assert_called_once_with()
    assert method.threshold_model is default.return_value


class WrongShapeThreshold(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, image: Tensor, phat: Tensor) -> Tensor:
        return self.value.sigmoid().expand(image.shape[0], 1)


def test_threshold_model_shape_checked() -> None:
    method = AdaptiveThresholding(
        nn.Conv2d(3, 1, 1), threshold_model=WrongShapeThreshold()
    )
    batch = {"input": torch.rand(2, 3, 8, 8), "target": torch.rand(2, 8, 8) > 0.5}
    with pytest.raises(ValueError, match=r"thresholds \[B\]=2, got \(2, 1\)"):
        method._training_step_network(batch)


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
