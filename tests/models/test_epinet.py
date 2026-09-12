# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for Epinet network modules."""

from typing import Any

import pytest
import torch
from torch import nn

from lightning_uq_box.models import (
    ConvEnsemblePriorFunction,
    EnsemblePriorFunction,
    Epinet,
    ProjectedMLP,
)


def _mlp_prior(**kwargs: Any) -> EnsemblePriorFunction:
    """Build a small MLP ensemble prior over a flattened input."""
    return EnsemblePriorFunction(
        n_inputs=12, n_outputs=5, num_ensemble=3, hidden_dims=[4], **kwargs
    )


def _conv_prior(**kwargs: Any) -> ConvEnsemblePriorFunction:
    """Build a small convolutional ensemble prior over images."""
    defaults: dict[str, Any] = {"input_size": 32} | kwargs
    return ConvEnsemblePriorFunction(
        in_channels=3, n_outputs=5, num_ensemble=3, **defaults
    )


def _prior_input(prior: nn.Module) -> torch.Tensor:
    """Return an input of the shape the given prior consumes."""
    return (
        torch.randn(2, 3, 32, 32)
        if isinstance(prior, ConvEnsemblePriorFunction)
        else torch.randn(2, 12)
    )


class TestProjectedMLP:
    def test_einsum_orientation_against_hand_computation(self) -> None:
        """Pin the reshape orientation with known weights and one-hot indices.

        A transposed ``[batch, index_dim, n_outputs]`` reshape still runs and still
        trains, so only an explicit hand-computed case rules it out. The final linear
        layer is set to emit ``[0, 1, ..., 7]`` regardless of input; reshaping that to
        ``[n_outputs=2, index_dim=4]`` gives rows ``[0, 1, 2, 3]`` and ``[4, 5, 6, 7]``,
        so contracting with the i-th one-hot index must return ``[i, i + 4]``.
        """
        model = ProjectedMLP(
            in_features=2, hidden_dims=[3], n_outputs=2, index_dim=4, concat_index=False
        )
        final = model.mlp[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.arange(8, dtype=torch.float))

        features = torch.zeros(1, 2)
        for i in range(4):
            index = torch.zeros(1, 4)
            index[0, i] = 1.0
            out = model(features, index)
            expected = torch.tensor([[float(i), float(i + 4)]])
            assert torch.allclose(out, expected), (
                f"one-hot index {i} gave {out.tolist()}, expected {expected.tolist()}. "
                "The reshape in ProjectedMLP.forward is likely transposed."
            )

    @pytest.mark.parametrize("concat_index", [True, False])
    def test_output_shape(self, concat_index: bool) -> None:
        model = ProjectedMLP(4, [8], 3, 6, concat_index=concat_index)
        assert model(torch.randn(7, 4), torch.randn(7, 6)).shape == (7, 3)

    def test_linear_in_the_index_without_concatenation(self) -> None:
        """Without concatenation the head is linear in z, so scaling z scales the output.

        With ``concat_index=True`` the index also enters the MLP, ahead of a ReLU, so
        the head is deliberately *not* linear in z; only the projection itself is.
        """
        model = ProjectedMLP(4, [8], 3, 6, concat_index=False)
        features = torch.randn(7, 4)
        index = torch.randn(7, 6)
        assert torch.allclose(
            model(features, 2.0 * index), 2.0 * model(features, index), atol=1e-5
        )


class TestEnsemblePriorFunctions:
    @pytest.mark.parametrize("build", [_mlp_prior, _conv_prior], ids=["mlp", "conv"])
    def test_output_shape_and_frozen(self, build) -> None:
        """A prior never trains, so none of its parameters require a gradient."""
        prior = build()
        out = prior(_prior_input(prior), torch.randn(2, 3))
        assert out.shape == (2, 5)
        assert all(not p.requires_grad for p in prior.parameters())

    @pytest.mark.parametrize("build", [_mlp_prior, _conv_prior], ids=["mlp", "conv"])
    def test_linear_in_the_index(self, build) -> None:
        """Members are combined linearly in z, so scaling z scales the output."""
        prior = build()
        x, index = _prior_input(prior), torch.randn(2, 3)
        assert torch.allclose(prior(x, 2.0 * index), 2.0 * prior(x, index), atol=1e-4)

    @pytest.mark.parametrize("build", [_mlp_prior, _conv_prior], ids=["mlp", "conv"])
    def test_seeded_members_are_independent_and_reproducible(self, build) -> None:
        """Each member is initialized separately, but the seed fixes the whole prior."""
        prior = build(seed=7)
        first, second = prior.members[0], prior.members[1]
        assert not torch.equal(next(first.parameters()), next(second.parameters())), (
            "ensemble members share weights, so they are not independently initialized"
        )

        x, index = _prior_input(prior), torch.randn(2, 3)
        torch.testing.assert_close(build(seed=7)(x, index), prior(x, index))

    def test_conv_prior_pads_and_resizes(self) -> None:
        """SAME padding fixes the read-out width, and other sizes are resized to it."""
        prior = _conv_prior()
        member = prior.members[0]
        assert isinstance(member, nn.Sequential)
        final = member[-1]
        assert isinstance(final, nn.Linear)
        assert final.in_features == 4 * 4 * 4
        assert prior(torch.randn(2, 3, 16, 16), torch.randn(2, 3)).shape == (2, 5)

    @pytest.mark.parametrize(
        "kwargs",
        [{"input_size": 0}, {"channels": []}, {"kernel_size": 0}, {"stride": 0}],
        ids=["input_size", "channels", "kernel_size", "stride"],
    )
    def test_conv_prior_rejects_invalid_geometry(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="positive"):
            _conv_prior(**kwargs)


class TestEpinetHead:
    def test_input_prior_skipped_when_scale_is_zero(self) -> None:
        """No prior over the raw input is built when it would be scaled away."""
        epinet = Epinet(
            n_feature_inputs=4, n_raw_inputs=2, n_outputs=2, input_prior_scale=0.0
        )
        assert epinet.input_prior is None

    def test_explicit_prior_is_used_and_frozen(self) -> None:
        """An explicit prior module is kept as given, frozen, and moves with the head."""
        conv_prior = ConvEnsemblePriorFunction(
            in_channels=3, n_outputs=10, num_ensemble=4, input_size=32
        )
        epinet = Epinet(
            n_feature_inputs=16,
            n_raw_inputs=0,
            n_outputs=10,
            index_dim=4,
            epi_prior_scale=4.0,
            input_prior_scale=1.0,
            input_prior=conv_prior,
        )
        assert epinet.input_prior is conv_prior
        assert all(not p.requires_grad for p in conv_prior.parameters())
        assert all(not p.requires_grad for p in epinet.epi_prior.parameters())
        assert all(p.requires_grad for p in epinet.train_epinet.parameters())

        out = epinet(torch.randn(2, 16), torch.randn(2, 3, 32, 32), torch.randn(2, 4))
        assert out.shape == (2, 10)

        # registered submodules follow .to(), unlike a plain attribute
        epinet = epinet.to(torch.float64)
        assert next(epinet.epi_prior.parameters()).dtype == torch.float64
        assert next(conv_prior.parameters()).dtype == torch.float64

    def test_priors_stay_in_eval_mode_when_the_head_trains(self) -> None:
        """Frozen priors must stay deterministic even while the epinet trains."""
        epinet = Epinet(
            n_feature_inputs=4, n_raw_inputs=2, n_outputs=2, input_prior_scale=0.3
        )
        epinet.train()
        assert epinet.train_epinet.training
        assert not epinet.epi_prior.training
        assert epinet.input_prior is not None
        assert not epinet.input_prior.training
