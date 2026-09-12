# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lightning_uq_box.models import HierarchicalProbUNet as Model
from lightning_uq_box.uq_methods import HierarchicalProbUNet
from lightning_uq_box.uq_methods.hierarchical_prob_unet import hierarchical_ce_loss


def make_method(**kwargs) -> HierarchicalProbUNet:
    classes = 1 if kwargs.get("task") == "binary" else 2
    return HierarchicalProbUNet(
        Model(
            encoder_name="resnet18",
            encoder_depth=3,
            encoder_weights=None,
            decoder_channels=(8, 4, 2),
            latent_dims=(1, 1),
            classes=classes,
        ),
        num_classes=classes,
        num_samples=2,
        **kwargs,
    )


@pytest.mark.parametrize("loss_type", ["geco", "elbo"])
@pytest.mark.parametrize("task", ["multiclass", "binary"])
def test_loss_prediction_and_gradients(loss_type: str, task: str) -> None:
    method = make_method(loss_type=loss_type, task=task)
    batch = {
        "input": torch.randn(2, 3, 32, 32),
        "target": torch.randint(2, (2, 32, 32)),
    }
    loss = method.compute_loss(batch)
    assert torch.isfinite(loss["loss"]) and loss["loss"] != 0
    assert {"kl_0", "kl_1", "kl_sum"} <= loss.keys()
    loss["loss"].backward()
    method.eval()
    with torch.no_grad():
        out = method.predict_step(batch["input"])
    assert out["logits"].shape == (2, method.num_classes, 32, 32, 2)
    assert torch.isfinite(out["pred_uct"]).all()
    assert method.ma_rec_loss.isnan()  # compute_loss has no mutation


def test_geco_direction_gradient_and_checkpoint(tmp_path: Path) -> None:
    method = make_method(decay=0.9, kappa=0.5, rate=0.01)
    rec = torch.tensor(10.0, requires_grad=True)
    constraint, ma = method._geco_constraint(rec, torch.tensor(2.0))
    constraint.backward()
    torch.testing.assert_close(rec.grad, torch.tensor(1.0))
    method._update_geco(constraint, ma)
    assert method.log_lagmul > 0
    previous = method.log_lagmul.clone()
    method._update_geco(torch.tensor(-2.0), torch.tensor(3.0))
    assert method.log_lagmul < previous
    rec2 = torch.tensor(1.0, requires_grad=True)
    constraint2, ma2 = method._geco_constraint(rec2, torch.tensor(2.0))
    torch.testing.assert_close(ma2, torch.tensor(2.8))
    constraint2.backward()
    torch.testing.assert_close(rec2.grad, torch.tensor(1.0))
    path = tmp_path / "state.pt"
    torch.save(method.state_dict(), path)
    restored = make_method()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.log_lagmul, method.log_lagmul)
    torch.testing.assert_close(restored.ma_rec_loss, method.ma_rec_loss)
    assert "log_lagmul" not in dict(method.named_parameters())


@pytest.mark.parametrize(
    "backbone,decoder", [(True, False), (False, True), (True, True)]
)
def test_freezing(backbone: bool, decoder: bool) -> None:
    method = make_method(freeze_backbone=backbone, freeze_decoder=decoder)
    for name, param in method.model.named_parameters():
        frozen = (backbone and "encoder" in name) or (
            decoder
            and (name.startswith(("segmentation_head", "prior_decoder.blocks.2")))
        )
        assert param.requires_grad != frozen, name


def test_soft_ce_and_batch_topk() -> None:
    logits = torch.tensor(
        [[[[0.0, 0.0]], [[0.0, 0.0]]], [[[0.0, 0.0]], [[2.0, 3.0]]]], requires_grad=True
    )
    target = torch.zeros_like(logits)
    target[:, 0] = 1
    all_pixels = hierarchical_ce_loss(logits, target, None)
    expected = -(target * logits.log_softmax(1)).sum(1)
    torch.testing.assert_close(all_pixels["sum"], expected.sum() / 2)
    torch.testing.assert_close(all_pixels["mean"], expected.mean())
    topk = hierarchical_ce_loss(logits, target, 0.5, True)
    torch.testing.assert_close(topk["mask"], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    topk["sum"].backward()
    assert logits.grad is not None and logits.grad[0].count_nonzero() == 0
    soft = torch.full_like(target, 0.5)
    torch.testing.assert_close(
        hierarchical_ce_loss(logits, soft, None)["sum"],
        F.cross_entropy(logits, soft, reduction="none").sum() / 2,
    )


def test_masking_and_stochastic_selection() -> None:
    logits = torch.zeros(2, 2, 4, 4, requires_grad=True)
    target = torch.full_like(logits, 0.5)
    valid = torch.zeros(2, 4, 4)
    valid[:, :2] = 1
    masks = []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        result = hierarchical_ce_loss(logits, target, 0.25, mask=valid)
        assert result["mask"].sum() == 8
        assert (result["mask"].reshape_as(valid) <= valid).all()
        masks.append(result["mask"])
    assert not torch.equal(*masks)
    empty = hierarchical_ce_loss(logits, target, mask=torch.zeros_like(valid))
    assert empty["sum"] == 0 and empty["mean"] == 0
    empty["sum"].backward()
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_geco_updates_only_training_step(monkeypatch: pytest.MonkeyPatch) -> None:
    method = make_method()
    monkeypatch.setattr(method, "log_dict", lambda *args, **kwargs: None)
    batch = {
        "input": torch.randn(2, 3, 32, 32),
        "target": torch.randint(2, (2, 32, 32)),
    }
    method.training_step(batch, 0).backward()
    assert method.ma_rec_loss.isfinite()
    saved = method.log_lagmul.clone(), method.ma_rec_loss.clone()
    method.eval()
    with torch.no_grad():
        method.validation_step(batch, 0)
    torch.testing.assert_close(saved[0], method.log_lagmul)
    torch.testing.assert_close(saved[1], method.ma_rec_loss)
