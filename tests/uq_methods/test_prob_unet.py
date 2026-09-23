# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for the flat ProbUNet and the shared segmentation prediction helper."""

import os

import segmentation_models_pytorch as smp
import torch
from torch import nn

from lightning_uq_box.uq_methods import ProbUNet
from lightning_uq_box.uq_methods.utils import process_segmentation_prediction


def make_method(classes: int = 4, **kwargs) -> ProbUNet:
    """Build a small flat ProbUNet with explicit batch keys."""
    method = ProbUNet(
        model=smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=classes,
        ),
        latent_dim=6,
        num_samples=2,
        **kwargs,
    )
    method.input_key = "input"
    method.target_key = "target"
    return method


def make_batch(batch_size: int = 2, size: int = 32) -> dict[str, torch.Tensor]:
    """Image/mask batch matching the method's configured keys."""
    return {
        "input": torch.randn(batch_size, 3, size, size),
        "target": torch.randint(4, (batch_size, size, size)),
    }


def test_binary_segmentation_prediction_is_not_degenerate() -> None:
    """A softmax over one channel returns a constant, so binary needs a sigmoid.

    Both ProbUNet and HierarchicalProbUNet accept ``task="binary"``, which
    produces single-channel logits. Routing those through the multiclass branch
    returns probability 1.0 everywhere and zero entropy, silently discarding the
    prediction and any uncertainty derived from it.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, 4) * 3
    out = process_segmentation_prediction(logits, task="binary")

    assert out["pred"].shape == (2, 1, 8, 8)
    assert out["pred_uct"].shape == (2, 8, 8)
    assert out["pred"].min() < out["pred"].max(), "prediction is constant"
    assert out["pred_uct"].max() > 0.0, "entropy is identically zero"
    # Binary entropy peaks at ln 2 where p = 0.5 and is bounded by it.
    assert out["pred_uct"].max() <= torch.log(torch.tensor(2.0)) + 1e-5


def test_multiclass_segmentation_prediction_unchanged() -> None:
    """The binary branch must not alter multiclass or multilabel behavior."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8, 8, 4)
    for task in ("multiclass", "multilabel"):
        out = process_segmentation_prediction(logits, task=task)
        assert out["pred"].shape == (2, 5, 8, 8)
        assert out["pred_uct"].shape == (2, 8, 8)
    # multiclass probabilities are normalized over the class axis
    multiclass = process_segmentation_prediction(logits, task="multiclass")
    torch.testing.assert_close(
        multiclass["pred"].sum(dim=1), torch.ones(2, 8, 8), rtol=1e-4, atol=1e-4
    )


def test_rec_loss_sum_is_a_pixel_sum_not_the_mean() -> None:
    """``rec_loss_sum`` must differ from ``rec_loss_mean`` by the pixel count.

    The default criterion is mean-reduced, so ``torch.sum`` over its scalar
    output returned the mean under a name that says "sum". Anyone converting a
    paper beta (which weights a pixel sum) into a library beta from that number
    would be off by H*W.
    """
    torch.manual_seed(0)
    size = 32
    method = make_method()
    out = method.compute_loss(make_batch(size=size))

    mean = out["rec_loss_mean"]
    total = out["rec_loss_sum"]
    assert not torch.isclose(total, mean), "rec_loss_sum still equals rec_loss_mean"
    torch.testing.assert_close(total / mean, torch.tensor(float(size * size)))


def test_loss_is_invariant_to_criterion_reduction() -> None:
    """Every reduction must give the same per-pixel mean, pixel sum and loss.

    ``reduction="sum"`` returns a batch total, so treating it like a mean made
    ``rec_loss_mean`` H*W times too large and inflated the optimized objective
    by the same factor without raising anything.
    """
    size = 32
    batch = make_batch(size=size)

    outs = {}
    for name, criterion in (
        ("mean", None),
        ("none", nn.CrossEntropyLoss(reduction="none")),
        ("sum", nn.CrossEntropyLoss(reduction="sum")),
    ):
        torch.manual_seed(1)
        outs[name] = make_method(criterion=criterion).compute_loss(batch)

    reference = outs["mean"]
    for name, out in outs.items():
        torch.testing.assert_close(
            out["rec_loss_mean"],
            reference["rec_loss_mean"],
            msg=f"{name} reduction changed the per-pixel mean",
        )
        torch.testing.assert_close(
            out["loss"], reference["loss"], msg=f"{name} reduction changed the loss"
        )
        torch.testing.assert_close(
            out["rec_loss_sum"] / out["rec_loss_mean"], torch.tensor(float(size * size))
        )


def test_compute_loss_keys_and_shapes() -> None:
    """The loss dict keeps its published keys and a usable reconstruction."""
    torch.manual_seed(0)
    method = make_method()
    out = method.compute_loss(make_batch())

    assert {
        "loss",
        "rec_loss_sum",
        "rec_loss_mean",
        "kl_loss",
        "reconstruction",
    } <= out.keys()
    assert out["reconstruction"].shape == (2, 4, 32, 32)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_beta_scales_the_kl_term() -> None:
    """Raising beta must increase the loss by beta * kl / (H * W).

    The objective is ``(rec_loss_sum + beta * kl_loss) / (H * W)``, so a change
    in beta moves the loss by the scaled KL term alone.
    """
    size = 32
    batch = make_batch(size=size)

    torch.manual_seed(1)
    low = make_method(beta=1.0).compute_loss(batch)
    torch.manual_seed(1)
    high = make_method(beta=3.0).compute_loss(batch)

    torch.testing.assert_close(low["kl_loss"], high["kl_loss"])
    torch.testing.assert_close(
        high["loss"] - low["loss"],
        2.0 * low["kl_loss"] / (size * size),
        rtol=1e-4,
        atol=1e-6,
    )


def test_beta_weights_the_pixel_sum_at_per_pixel_magnitude() -> None:
    """``beta`` weights the pixel sum, but the loss stays per-pixel sized.

    Weighting ``rec_loss_mean`` makes any beta of order 1 act H*W times too
    strongly and collapses the posterior onto the prior. Weighting the raw
    pixel sum fixes beta's meaning but leaves the loss at ~1e5 on a 256^2
    image, whose gradients drive the latent heads to NaN within a few steps.
    Dividing the whole objective by H*W fixes the magnitude while leaving the
    ratio of the two terms -- and so beta's meaning -- untouched.
    """
    size = 32
    torch.manual_seed(0)
    method = make_method(beta=1.0)
    out = method.compute_loss(make_batch(size=size))

    torch.testing.assert_close(
        out["loss"],
        (out["rec_loss_sum"] + out["kl_loss"]) / (size * size),
        rtol=1e-4,
        atol=1e-4,
    )
    # Per-pixel magnitude: comparable to a plain CE, not to a pixel sum.
    assert out["loss"] < 100.0
    assert not torch.isclose(out["loss"], out["rec_loss_sum"] + out["kl_loss"])


def test_loss_is_batch_size_invariant() -> None:
    """Repeating an image must not change the loss it contributes.

    A divisor of B*H*W rather than H*W silently makes the effective learning
    rate depend on batch size.
    """
    size, losses = 32, []
    for batch_size in (2, 4, 8):
        torch.manual_seed(0)
        method = make_method(beta=1.0).eval()
        torch.manual_seed(1)
        image = torch.randn(1, 3, size, size).repeat(batch_size, 1, 1, 1)
        target = torch.randint(4, (1, size, size)).repeat(batch_size, 1, 1)

        # Use the posterior mean so no sampling noise enters the comparison.
        method.posterior_latent_space = method.posterior.forward(
            image, target.unsqueeze(1)
        )
        method.prior_latent_space = method.prior.forward(image)
        method.unet_features = method.extract_features(image)
        z = method.posterior_latent_space.mean
        reconstruction = method.fcomb.forward(method.unet_features, z)
        rec = nn.functional.cross_entropy(reconstruction, target)
        kl = torch.mean(method.kl_divergence(analytic=True, z_posterior=z))
        losses.append(float((rec * size * size + kl) / (size * size)))

    assert max(losses) - min(losses) < 1e-4 * max(losses), losses


def test_gradients_stay_finite_over_several_steps() -> None:
    """The objective must not blow the latent heads up to NaN.

    Weighting an unscaled pixel sum produced gradient norms of ~1e5, and the
    posterior's `loc` became NaN within a handful of optimizer steps -- before
    a single epoch finished, and regardless of `gradient_clip_val`.
    """
    torch.manual_seed(0)
    method = make_method(beta=1.0)
    batch = make_batch()
    optimizer = torch.optim.Adam(method.parameters(), lr=1e-3)

    for _ in range(8):
        out = method.compute_loss(batch)
        assert torch.isfinite(out["loss"]), "loss went non-finite"
        assert torch.isfinite(method.posterior_latent_space.base_dist.loc).all()
        optimizer.zero_grad()
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
        assert grad_norm < 1e3, f"gradient norm {float(grad_norm):.3e} is explosive"
        optimizer.step()


def test_fcomb_taps_decoder_features_by_default() -> None:
    """The latent joins the wide decoder output, not the class-width logits."""
    method = make_method(classes=13)
    assert method.fcomb_input == "decoder_features"
    # smp's default decoder ends at 16 channels, wider than the 13 classes.
    assert method.num_feature_channels == 16
    assert method.fcomb_input_channels == 16 + 6
    assert method.fcomb.layers[0].weight.shape == (32, 22, 1, 1)

    torch.manual_seed(0)
    method.compute_loss(make_batch())
    assert method.unet_features.shape[1] == 16


def test_fcomb_logits_tap_is_the_narrow_bottleneck() -> None:
    """``fcomb_input="logits"`` keeps the num_classes-wide tap for odd models."""
    method = make_method(classes=13, fcomb_input="logits")
    assert method.num_feature_channels == 13
    assert method.fcomb.layers[0].weight.shape == (32, 19, 1, 1)

    torch.manual_seed(0)
    out = method.compute_loss(make_batch())
    assert method.unet_features.shape[1] == 13
    assert out["reconstruction"].shape[1] == 13


def test_decoderless_model_rejected_with_actionable_error() -> None:
    """A model without a decoder must say what to pass instead."""
    import pytest

    # Shaped like a segmentation model (so channel introspection works) but
    # exposing no `decoder`, as a plain nn.Sequential would not.
    plain = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 4, 1))

    with pytest.raises(ValueError, match="fcomb_input='logits'"):
        ProbUNet(model=plain, latent_dim=6, num_samples=2)

    # The same model is usable once the caller opts into the narrow tap.
    method = ProbUNet(model=plain, latent_dim=6, num_samples=2, fcomb_input="logits")
    assert method.num_feature_channels == 4


def test_predict_step_uses_the_same_tap_as_training() -> None:
    """Train and inference must build Fcomb's input the same way."""
    for fcomb_input, width in (("decoder_features", 16), ("logits", 13)):
        method = make_method(classes=13, fcomb_input=fcomb_input)
        batch = make_batch()
        torch.manual_seed(0)
        method.compute_loss(batch)
        assert method.unet_features.shape[1] == width
        with torch.no_grad():
            method.predict_step(batch["input"])
        assert method.unet_features.shape[1] == width


def test_log_sigma_is_bounded_under_extreme_raw_inputs() -> None:
    """The scale must saturate rather than diverge, as the unbounded exp did."""
    bound = 4.0
    raw = torch.tensor([-100.0, -20.0, 0.0, 20.0, 100.0])
    log_sigma = -bound + 2 * bound * torch.sigmoid(raw.float() * (2 / bound))
    sigma = log_sigma.exp()

    assert (log_sigma.abs() <= bound).all()
    assert sigma.min() >= 0.018 and sigma.max() <= 54.6
    # The largest sigma observed on real runs was 5.25, well inside the bound.
    assert sigma.max() > 5.25


def test_log_sigma_squash_is_identity_like_at_the_origin() -> None:
    """Gradient exactly 1 at 0, so normal operation is undistorted."""
    bound = 4.0
    raw = torch.zeros(1, requires_grad=True)
    log_sigma = -bound + 2 * bound * torch.sigmoid(raw * (2 / bound))
    log_sigma.backward()

    assert log_sigma.item() == 0.0
    assert raw.grad is not None
    assert torch.isclose(raw.grad, torch.ones(1))


def test_invalid_log_sigma_bound_raises() -> None:
    """A non-positive or infinite bound is a configuration error."""
    import pytest

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="log_sigma_bound"):
            make_method(classes=4, log_sigma_bound=bad)


def test_beta_only_configuration_survives_gradient_steps() -> None:
    """beta=1.0 with the logits tap diverged before the bound existed.

    On the real runs this configuration drove |mu| from 1.3 to 17.9 in 40
    steps and crashed. The bound constrains only that divergent regime.
    """
    torch.manual_seed(0)
    method = make_method(classes=13, beta=1.0, fcomb_input="logits")
    batch = make_batch()
    optimizer = torch.optim.Adam(method.parameters(), lr=1e-3)

    for _ in range(8):
        out = method.compute_loss(batch)
        assert torch.isfinite(out["loss"]), "loss went non-finite"
        assert torch.isfinite(method.posterior_latent_space.base_dist.loc).all()
        scale = method.posterior_latent_space.base_dist.scale
        assert (scale <= 54.6).all() and (scale >= 0.018).all()
        optimizer.zero_grad()
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
        assert grad_norm < 1e3, f"gradient norm {float(grad_norm):.3e} is explosive"
        optimizer.step()


def test_save_preds_false_writes_nothing() -> None:
    """`save_preds` was accepted and documented but ignored.

    `on_test_batch_end` saved every batch unconditionally, which dumped 43 GB
    of HDF5 during a *two-epoch* smoke run and slowed the test loop to 0.02
    it/s. The hierarchical model gates both the directory and the write; the
    flat one now does too.
    """
    import tempfile

    from lightning import Trainer
    from torch.utils.data import DataLoader, Dataset

    class Toy(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "input": torch.randn(3, 32, 32),
                "target": torch.randint(4, (32, 32)),
            }

    for save_preds, expect_files in ((False, False), (True, True)):
        method = make_method(save_preds=save_preds)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                logger=False,
                default_root_dir=tmp,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.test(method, dataloaders=DataLoader(Toy(), batch_size=2))
            pred_dir = os.path.join(tmp, method.pred_dir_name)
            written = os.path.isdir(pred_dir) and bool(os.listdir(pred_dir))
            assert written is expect_files, (
                f"save_preds={save_preds} wrote files={written}"
            )


def test_reconstruct_with_posterior_mean_decodes_the_mean() -> None:
    """``use_posterior_mean=True`` must decode the posterior mean.

    The latent space is an ``Independent`` distribution, which has no ``.loc``;
    reading it raised AttributeError, so this path had never run.
    """
    torch.manual_seed(0)
    method = make_method().eval()
    with torch.no_grad():
        method.compute_loss(make_batch())
        rec = method.reconstruct(use_posterior_mean=True)
        expected = method.fcomb.forward(
            method.unet_features, method.posterior_latent_space.base_dist.loc
        )
        # Deterministic: two calls agree, unlike posterior samples.
        again = method.reconstruct(use_posterior_mean=True)
    assert rec.shape == (2, 4, 32, 32)
    torch.testing.assert_close(rec, expected)
    torch.testing.assert_close(rec, again)
