# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

import math

import pytest
import torch
from torch.distributions import kl_divergence

from lightning_uq_box.models import (
    HierarchicalProbUNet,
    HierarchicalUnetDecoder,
    PreActResBlock,
)


@pytest.fixture
def model() -> HierarchicalProbUNet:
    torch.manual_seed(17)
    return HierarchicalProbUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        decoder_channels=(16, 8, 4, 4, 2),
        latent_dims=(2, 2),
        decoder_use_norm=False,
    )


@pytest.mark.parametrize("latent_dims", [(1, 1, 1, 1), (2, 2)])
@pytest.mark.parametrize("encoder", ["resnet18", "tu-mobilenetv3_small_100"])
def test_scales_and_encoder_swap(latent_dims: tuple[int, ...], encoder: str) -> None:
    model = HierarchicalProbUNet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=1,
        latent_dims=latent_dims,
        decoder_channels=(16, 8, 4, 4, 2),
    ).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        p = model.prior_decoder(model.prior_encoder(x))
        forced = model.prior_decoder(model.prior_encoder(x), z_q=p.used_latents)
        logits, q, prior = model.reconstruct(x, torch.randn(2, 2, 64, 64))
    assert logits.shape == (2, 2, 64, 64)
    for i, (dim, dist, z) in enumerate(
        zip(latent_dims, p.distributions, p.used_latents, strict=True)
    ):
        assert dist.event_shape == (dim,)
        assert dist.batch_shape == (2, 2 * 2**i, 2 * 2**i)
        assert z.shape == (2, dim, 2 * 2**i, 2 * 2**i)
        torch.testing.assert_close(z, forced.used_latents[i], rtol=0, atol=0)
        assert (kl_divergence(q[i], prior[i]) >= 0).all()
        channels = list(model.prior_encoder.out_channels[1:])[::-1]
        base = channels[0] if i == 0 else (16, 8, 4, 4, 2)[i - 1]
        conv1 = model.prior_decoder.blocks[i].conv1
        assert isinstance(conv1, torch.nn.Sequential)
        conv = conv1[0]
        assert isinstance(conv, torch.nn.Conv2d)
        assert conv.in_channels == base + channels[i + 1] + dim


def test_matching_conditionals_have_zero_kl() -> None:
    q_decoder = HierarchicalUnetDecoder(
        encoder_channels=(3, 4, 8, 16),
        decoder_channels=(8, 4, 2),
        n_blocks=3,
        latent_dims=(2, 2),
        use_norm=False,
    )
    p_decoder = HierarchicalUnetDecoder(
        encoder_channels=(3, 4, 8, 16),
        decoder_channels=(8, 4, 2),
        n_blocks=3,
        latent_dims=(2, 2),
        use_norm=False,
    )
    p_decoder.load_state_dict(q_decoder.state_dict())
    features = [torch.randn(2, c, s, s) for c, s in [(3, 32), (4, 16), (8, 8), (16, 4)]]
    q = q_decoder(features)
    p = p_decoder(features, z_q=q.used_latents)
    for qi, pi, zq, zp in zip(
        q.distributions, p.distributions, q.used_latents, p.used_latents, strict=True
    ):
        assert kl_divergence(qi, pi).abs().max() < 1e-5
        assert zq is zp
    # A free prior draw must change the finer conditional, guarding the test itself.
    free = p_decoder(features)
    assert not torch.equal(free.distributions[1].mean, q.distributions[1].mean)


def test_reconstruction_reparameterizes_posterior(model: HierarchicalProbUNet) -> None:
    x = torch.randn(2, 3, 64, 64)
    logits, _, _ = model.reconstruct(x, torch.randn(2, 2, 64, 64))
    logits.square().mean().backward()
    # Prior gradients alone would also pass if the posterior used sample().
    for encoder in [model.prior_encoder, model.posterior_encoder]:
        grad = next(encoder.parameters()).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for head in model.posterior_decoder.latent_heads:
        assert isinstance(head, torch.nn.Conv2d)
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0


def test_all_trainable_parameters_used(model: HierarchicalProbUNet) -> None:
    logits, q, p = model.reconstruct(
        torch.randn(2, 3, 64, 64), torch.randn(2, 2, 64, 64)
    )
    (
        logits.square().mean()
        + sum(kl_divergence(qi, pi).mean() for qi, pi in zip(q, p, strict=True))
    ).backward()
    assert not [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]


def test_stateless_means_and_intervention(model: HierarchicalProbUNet) -> None:
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = model.sample(x, mean=True)
        model.sample(torch.randn_like(x))
        torch.testing.assert_close(expected, model.sample(x, mean=[True, True]))
        assert not torch.equal(expected, model.sample(x, mean=[True, False]))
        prior = model.prior_decoder(model.prior_encoder(x), mean=True)
        torch.testing.assert_close(expected, model.sample(x, z_q=prior.used_latents))
    with pytest.raises(ValueError, match="per scale"):
        model.sample(x, mean=[True])
    with pytest.raises(ValueError, match="shape"):
        model.sample(x, z_q=[torch.zeros(1), torch.zeros(1)])


def test_invalid_hierarchy() -> None:
    with pytest.raises(ValueError, match="stitching"):
        HierarchicalProbUNet(
            encoder_depth=2,
            encoder_weights=None,
            decoder_channels=(8, 4),
            latent_dims=(1, 1),
        )


def test_residual_projection() -> None:
    block = PreActResBlock(3, 5, down_channels=2)
    assert block(torch.randn(2, 3, 8, 8)).shape == (2, 5, 8, 8)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in block.modules())


def test_extreme_scales_remain_finite_under_autocast(
    model: HierarchicalProbUNet,
) -> None:
    model.eval()
    with torch.no_grad():
        for decoder in [model.prior_decoder, model.posterior_decoder]:
            for head in decoder.latent_heads:
                assert isinstance(head, torch.nn.Conv2d)
                assert head.bias is not None
                head.weight.zero_()
                head.bias[:2].zero_()
                head.bias[2:].fill_(1000)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, q, p = model.reconstruct(
                torch.randn(2, 3, 64, 64), torch.randn(2, 2, 64, 64)
            )
    assert torch.isfinite(logits).all()
    for qi, pi in zip(q, p, strict=True):
        assert qi.mean.dtype == torch.float32
        assert torch.isfinite(kl_divergence(qi, pi)).all()


def scale_decoder(bound: float, bias: float) -> HierarchicalUnetDecoder:
    decoder = HierarchicalUnetDecoder(
        encoder_channels=(3, 4, 8, 16),
        decoder_channels=(8, 4, 2),
        n_blocks=3,
        latent_dims=(1, 1),
        use_norm=False,
        log_sigma_bound=bound,
    )
    with torch.no_grad():
        for head in decoder.latent_heads:
            assert isinstance(head, torch.nn.Conv2d) and head.bias is not None
            head.weight.zero_()
            head.bias.zero_()
            head.bias[1:].fill_(bias)
    return decoder


def scale_features() -> list[torch.Tensor]:
    return [torch.ones(2, c, s, s) for c, s in [(3, 32), (4, 16), (8, 8), (16, 4)]]


def test_log_sigma_bound_limits_kl() -> None:
    values = []
    for bound in (4.0, 10.0):
        q = scale_decoder(bound, 20)(scale_features(), mean=True)
        p = scale_decoder(bound, -20)(scale_features(), z_q=q.used_latents)
        values.append(
            torch.stack(
                [
                    kl_divergence(qi, pi).mean()
                    for qi, pi in zip(q.distributions, p.distributions, strict=True)
                ]
            )
        )
        # With equal means, KL <= (exp(4*bound) - 1 - 4*bound)/2.
        ceiling = (torch.exp(torch.tensor(4 * bound)) - 1 - 4 * bound) / 2
        assert (values[-1] <= ceiling).all()
    assert (values[1] > values[0] * 1e6).all()


def test_log_sigma_bound_preserves_recovery_gradient() -> None:
    torch.manual_seed(42)
    decoder = scale_decoder(4.0, -20.0)
    out = decoder(scale_features(), mean=True)
    sum(d.base_dist.scale.log().sum() for d in out.distributions).backward()
    for head in decoder.latent_heads:
        assert isinstance(head, torch.nn.Conv2d)
        assert head.weight.grad is not None
        # Only scale rows: mean gradients cannot disguise a clamp trap.
        assert head.weight.grad[1:].abs().sum() > 0
        assert torch.isfinite(head.weight.grad).all()


@pytest.mark.parametrize("bias", [-0.5, 0.0, 0.5])
def test_log_sigma_bound_is_near_identity_in_healthy_range(bias: float) -> None:
    decoder = scale_decoder(4.0, bias)
    for dist in decoder(scale_features(), mean=True).distributions:
        torch.testing.assert_close(
            dist.base_dist.scale,
            torch.full_like(dist.base_dist.scale, math.exp(bias)),
            rtol=0.02,
            atol=0,
        )


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_log_sigma_bound(bound: float) -> None:
    with pytest.raises(ValueError, match="log_sigma_bound"):
        scale_decoder(bound, 0)
    with pytest.raises(ValueError, match="log_sigma_bound"):
        HierarchicalProbUNet(encoder_weights=None, log_sigma_bound=bound)


@pytest.mark.parametrize("factor", [1, 4])
def test_pooled_latent_conditioning_and_gradients(factor: int) -> None:
    torch.manual_seed(42)
    model = HierarchicalProbUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        decoder_channels=(16, 8, 4, 4, 2),
        decoder_use_norm=False,
        latent_pool_factor=factor,
    ).eval()
    x = torch.randn(2, 1, 128, 128)
    features = model.prior_encoder(x)
    q = model.prior_decoder(features, mean=True)
    p = model.prior_decoder(features, z_q=q.used_latents)
    for i, (qi, pi) in enumerate(zip(q.distributions, p.distributions, strict=True)):
        size = 4 * 2**i // factor
        assert qi.batch_shape == (2, size, size)
        torch.testing.assert_close(
            kl_divergence(qi, pi), torch.zeros_like(qi.mean[..., 0])
        )
        assert p.used_latents[i] is q.used_latents[i]
    changed = list(q.used_latents)
    changed[0] = changed[0] + 1
    intervention = model.prior_decoder(features, z_q=changed)
    assert not torch.equal(intervention.distributions[1].mean, q.distributions[1].mean)
    logits, posterior, prior = model.reconstruct(x, torch.randn(2, 2, 128, 128))
    assert logits.shape == (2, 2, 128, 128)
    loss = logits.square().mean() + sum(
        kl_divergence(qi, pi).mean() for qi, pi in zip(posterior, prior, strict=True)
    )
    loss.backward()
    for head in model.posterior_decoder.latent_heads:
        assert isinstance(head, torch.nn.Conv2d)
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_latent_pool_factor(factor: int) -> None:
    with pytest.raises(ValueError, match="latent_pool_factor"):
        HierarchicalUnetDecoder(
            encoder_channels=(3, 4, 8, 16),
            decoder_channels=(8, 4, 2),
            n_blocks=3,
            latent_dims=(1, 1),
            latent_pool_factor=factor,
        )
