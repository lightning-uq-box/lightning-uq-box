# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test joint prediction metrics."""

import math

import pytest
import torch

from lightning_uq_box.eval_utils import (
    average_sampled_log_likelihood,
    categorical_kl,
    dyadic_batch_indices,
    joint_log_likelihood,
    joint_log_loss_dyadic,
    marginal_log_likelihood,
    marginal_log_loss,
)


class TestAverageSampledLogLikelihood:
    def test_matches_log_mean_exp(self) -> None:
        lls = torch.tensor([-1.0, -2.0, -3.0])
        expected = torch.logsumexp(lls, dim=0) - math.log(3)
        assert torch.allclose(average_sampled_log_likelihood(lls), expected)

    def test_all_neg_inf_returns_neg_inf(self) -> None:
        lls = torch.full((4,), float("-inf"))
        out = average_sampled_log_likelihood(lls)
        assert torch.isneginf(out)
        assert not torch.isnan(out)

    def test_partial_neg_inf_stays_finite(self) -> None:
        lls = torch.tensor([float("-inf"), -1.0])
        out = average_sampled_log_likelihood(lls)
        assert torch.isfinite(out)
        assert torch.allclose(out, torch.tensor(-1.0 - math.log(2), dtype=out.dtype))


class TestJointLogLikelihood:
    def test_tau_one_equals_marginal(self) -> None:
        """With a single input there is nothing to be joint about."""
        torch.manual_seed(0)
        logits = torch.randn(7, 1, 4)
        targets = torch.tensor([2])

        joint = joint_log_likelihood(logits, targets)
        marginal = marginal_log_likelihood(logits, targets)

        assert torch.allclose(joint, marginal, atol=1e-10)

    def test_single_sample_joint_equals_sum_of_marginals(self) -> None:
        """A one-sample ENN is a plain deterministic predictor."""
        torch.manual_seed(1)
        logits = torch.randn(1, 5, 3)
        targets = torch.tensor([0, 1, 2, 1, 0])

        joint = joint_log_likelihood(logits, targets)
        marginal = marginal_log_likelihood(logits, targets)

        assert torch.allclose(joint, marginal, atol=1e-10)

    def test_correlated_enn_beats_independent_marginals(self) -> None:
        """The point of the joint metric.

        Two ENN samples that are perfectly confident in opposite directions have the
        same marginal distribution as a single 50/50 predictor, but on a repeated
        input the correlated ENN pays the log-loss once, while the independent
        predictor pays it tau times.
        """
        tau = 6
        # sample 0 always says class 0, sample 1 always says class 1
        correlated = torch.zeros(2, tau, 2)
        correlated[0, :, 0] = 10.0
        correlated[1, :, 1] = 10.0

        # a single predictor with the same marginal (50/50 on every input)
        independent = torch.zeros(1, tau, 2)

        targets = torch.zeros(tau, dtype=torch.long)

        joint_corr = joint_log_likelihood(correlated, targets)
        joint_indep = joint_log_likelihood(independent, targets)

        # marginals agree: both give probability 1/2 to class 0 on each input
        assert torch.allclose(
            marginal_log_likelihood(correlated, targets),
            marginal_log_likelihood(independent, targets),
            atol=1e-3,
        )
        # but the joint strongly prefers the correlated predictor
        assert joint_corr > joint_indep
        # the correlated ENN pays roughly log(2) once, the independent one tau times
        assert joint_corr.item() == pytest.approx(-math.log(2), abs=1e-3)
        assert joint_indep.item() == pytest.approx(-tau * math.log(2), abs=1e-3)

    def test_rejects_wrong_shapes(self) -> None:
        with pytest.raises(ValueError, match="num_samples, tau, num_classes"):
            joint_log_likelihood(torch.randn(3, 4), torch.tensor([0]))
        with pytest.raises(ValueError, match="to match the tau axis"):
            joint_log_likelihood(torch.randn(2, 3, 4), torch.tensor([0]))


class TestMarginalLogLikelihood:
    def test_averages_probabilities_not_logits(self) -> None:
        """Hand-computed: two samples, one input, two classes."""
        logits = torch.tensor([[[0.0, float("-inf")]], [[float("-inf"), 0.0]]])
        targets = torch.tensor([0])
        # mean probability of class 0 is (1 + 0) / 2 = 0.5
        out = marginal_log_likelihood(logits, targets)
        assert out.item() == pytest.approx(-math.log(2), abs=1e-10)


class TestDyadicBatchIndices:
    def test_shape_and_distinct_count(self) -> None:
        generator = torch.Generator().manual_seed(0)
        idx = dyadic_batch_indices(100, tau=10, kappa=2, generator=generator)
        assert idx.shape == (10,)
        assert len(torch.unique(idx)) <= 2
        assert int(idx.min()) >= 0
        assert int(idx.max()) < 100

    def test_reproducible_with_generator(self) -> None:
        a = dyadic_batch_indices(50, generator=torch.Generator().manual_seed(3))
        b = dyadic_batch_indices(50, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("num_data,kappa", [(0, 2), (10, 0)])
    def test_rejects_invalid_arguments(self, num_data: int, kappa: int) -> None:
        with pytest.raises(ValueError):
            dyadic_batch_indices(num_data, kappa=kappa)


class TestJointLogLossDyadic:
    def test_correlated_enn_has_lower_loss(self) -> None:
        num_data = 20
        correlated = torch.zeros(2, num_data, 2)
        correlated[0, :, 0] = 10.0
        correlated[1, :, 1] = 10.0
        independent = torch.zeros(1, num_data, 2)
        targets = torch.zeros(num_data, dtype=torch.long)

        gen_a = torch.Generator().manual_seed(0)
        gen_b = torch.Generator().manual_seed(0)
        loss_corr = joint_log_loss_dyadic(
            correlated, targets, tau=10, kappa=2, num_batches=20, generator=gen_a
        )
        loss_indep = joint_log_loss_dyadic(
            independent, targets, tau=10, kappa=2, num_batches=20, generator=gen_b
        )
        assert loss_corr < loss_indep

    def test_tau_one_matches_marginal_log_loss(self) -> None:
        torch.manual_seed(2)
        logits = torch.randn(4, 8, 3)
        targets = torch.randint(0, 3, (8,))
        # with tau=1 the joint log loss of one input is its marginal log loss
        for i in range(8):
            joint = joint_log_loss_dyadic(
                logits[:, i : i + 1, :],
                targets[i : i + 1],
                tau=1,
                kappa=1,
                num_batches=1,
            )
            marginal = marginal_log_loss(logits[:, i : i + 1, :], targets[i : i + 1])
            assert torch.allclose(joint, marginal, atol=1e-10)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="num_samples, num_data, num_classes"):
            joint_log_loss_dyadic(torch.randn(3, 4), torch.tensor([0, 1, 2, 0]))


class TestCategoricalKL:
    def test_zero_for_identical(self) -> None:
        p = torch.tensor([0.2, 0.3, 0.5])
        assert categorical_kl(p, p).item() == pytest.approx(0.0, abs=1e-10)

    def test_matches_hand_computation(self) -> None:
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.25, 0.75])
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        assert categorical_kl(p, q).item() == pytest.approx(expected, abs=1e-10)

    def test_batched(self) -> None:
        p = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
        q = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        out = categorical_kl(p, q)
        assert out.shape == (2,)
        assert out[0].item() == pytest.approx(0.0, abs=1e-10)
        assert out[1].item() > 0.0


@pytest.mark.parametrize("metric", [joint_log_likelihood, marginal_log_likelihood])
def test_extreme_logits_stay_in_log_domain(metric) -> None:
    logits = torch.tensor([[[0.0, -10000.0]], [[0.0, -10002.0]]])
    result = metric(logits, torch.tensor([1]))
    expected = torch.logsumexp(
        torch.tensor([-10000.0, -10002.0], dtype=torch.float64), 0
    ) - math.log(2)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "kwargs", [{"tau": 0}, {"num_batches": 0}, {"num_batches": -1}]
)
def test_dyadic_rejects_empty_averages(kwargs) -> None:
    with pytest.raises(ValueError):
        joint_log_loss_dyadic(
            torch.zeros(2, 3, 2), torch.zeros(3, dtype=torch.long), **kwargs
        )


def test_kl_respects_zero_probability_events() -> None:
    assert (
        categorical_kl(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])).item() == 0
    )
    assert torch.isposinf(
        categorical_kl(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    )


class TestArgumentValidation:
    """Cover the guards that reject malformed inputs.

    Every metric here validates its arguments before touching the data, because a
    silently broadcast or transposed tensor yields a plausible-looking number rather
    than an error. These tests pin each guard so a refactor cannot quietly drop one.
    """

    def test_average_sampled_log_likelihood_rejects_non_vectors(self) -> None:
        with pytest.raises(ValueError, match="nonempty vector"):
            average_sampled_log_likelihood(torch.randn(2, 3))
        with pytest.raises(ValueError, match="nonempty vector"):
            average_sampled_log_likelihood(torch.empty(0))

    @pytest.mark.parametrize("metric", [joint_log_likelihood, marginal_log_likelihood])
    def test_likelihoods_reject_empty_axes(self, metric) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            metric(torch.randn(0, 2, 3), torch.zeros(2, dtype=torch.long))
        with pytest.raises(ValueError, match="nonempty"):
            metric(torch.randn(2, 2, 0), torch.zeros(2, dtype=torch.long))

    def test_marginal_log_likelihood_rejects_wrong_shapes(self) -> None:
        with pytest.raises(ValueError, match="num_samples, tau, num_classes"):
            marginal_log_likelihood(torch.randn(3, 4), torch.tensor([0]))
        with pytest.raises(ValueError, match="to match the tau axis"):
            marginal_log_likelihood(torch.randn(2, 3, 4), torch.tensor([0]))

    def test_dyadic_batch_indices_rejects_nonpositive_tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be at least 1"):
            dyadic_batch_indices(10, tau=0)

    def test_joint_log_loss_dyadic_rejects_empty_axes(self) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            joint_log_loss_dyadic(
                torch.randn(2, 3, 0), torch.zeros(3, dtype=torch.long)
            )

    def test_joint_log_loss_dyadic_rejects_mismatched_targets(self) -> None:
        with pytest.raises(ValueError, match=r"targets must have shape"):
            joint_log_loss_dyadic(
                torch.randn(2, 3, 4), torch.zeros(5, dtype=torch.long)
            )

    def test_categorical_kl_rejects_negative_eps(self) -> None:
        p = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="eps must be nonnegative"):
            categorical_kl(p, p, eps=-1e-9)
