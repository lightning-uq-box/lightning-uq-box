# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Joint prediction metrics for epistemic neural networks.

Ports the joint and marginal log-likelihood metrics and the dyadic sampling scheme of

* https://arxiv.org/abs/2107.08924

An ENN produces a set of ``num_samples`` logit vectors per input, one per draw of the
epistemic index. A *marginal* metric averages the predictive distribution over those
samples before scoring each input independently; a *joint* metric keeps the samples
tied together across a batch of ``tau`` inputs, which is what distinguishes a model
that has captured the dependence between its predictions from one that has not.

All functions here are pure tensor functions with no Lightning dependency, matching
:mod:`lightning_uq_box.eval_utils.uq_computation`.
"""

import torch
from torch import Tensor


def average_sampled_log_likelihood(lls: Tensor) -> Tensor:
    """Average log-likelihoods over ENN samples in a numerically safe way.

    Computes ``log(mean(exp(lls)))`` over the sample axis. When any sample assigns
    zero probability (``-inf`` log-likelihood) the plain log-mean-exp is still finite,
    but if *every* sample does, the result is ``-inf``, which is returned directly
    rather than produced through a ``nan``-generating subtraction. This mirrors the
    guard in the reference implementation.

    Args:
        lls: log-likelihoods of shape [num_samples]

    Returns:
        scalar averaged log-likelihood
    """
    if torch.all(torch.isneginf(lls)):
        return torch.tensor(float("-inf"), dtype=lls.dtype, device=lls.device)
    num_samples = torch.tensor(lls.shape[0], dtype=lls.dtype, device=lls.device)
    return torch.logsumexp(lls, dim=0) - torch.log(num_samples)


def joint_log_likelihood(logits: Tensor, targets: Tensor) -> Tensor:
    r"""Joint log-likelihood of a batch of targets under an ENN.

    Sums the per-input log-probabilities *within* each index sample, so the samples
    stay tied across the ``tau`` inputs, then averages the resulting likelihoods over
    samples with a log-mean-exp. This is :math:`\ln \hat{P}_{1:\tau}(y_{1:\tau})` of
    equation 2 in the paper.

    Args:
        logits: ENN logits of shape [num_samples, tau, num_classes]
        targets: class indices of shape [tau]

    Returns:
        scalar joint log-likelihood
    """
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits of shape [num_samples, tau, num_classes], "
            f"got shape {tuple(logits.shape)}."
        )
    if targets.shape[0] != logits.shape[1]:
        raise ValueError(
            f"Expected {logits.shape[1]} targets to match the tau axis of logits, "
            f"got {targets.shape[0]}."
        )

    log_probs = torch.log_softmax(logits.double(), dim=-1)
    # [num_samples, tau] -> log-prob of the observed class under each sample
    target_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=targets.long().view(1, -1, 1).expand(log_probs.shape[0], -1, 1),
    ).squeeze(-1)
    # sum over the tau inputs *inside* each sample, then average over samples
    return average_sampled_log_likelihood(target_log_probs.sum(dim=-1))


def marginal_log_likelihood(logits: Tensor, targets: Tensor) -> Tensor:
    """Marginal log-likelihood of a batch of targets under an ENN.

    Averages the *probabilities* over index samples first, collapsing the ENN to a
    single predictive distribution per input, and only then scores each input. The
    result is the sum of independent per-input log-probabilities, and therefore carries
    no information about how the predictions covary.

    Args:
        logits: ENN logits of shape [num_samples, tau, num_classes]
        targets: class indices of shape [tau]

    Returns:
        scalar marginal log-likelihood, summed over the tau inputs
    """
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits of shape [num_samples, tau, num_classes], "
            f"got shape {tuple(logits.shape)}."
        )
    if targets.shape[0] != logits.shape[1]:
        raise ValueError(
            f"Expected {logits.shape[1]} targets to match the tau axis of logits, "
            f"got {targets.shape[0]}."
        )

    probs = torch.softmax(logits.double(), dim=-1).mean(dim=0)  # [tau, num_classes]
    target_probs = torch.gather(
        probs, dim=-1, index=targets.long().view(-1, 1)
    ).squeeze(-1)
    return torch.log(target_probs).sum()


def dyadic_batch_indices(
    num_data: int,
    tau: int = 10,
    kappa: int = 2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw the indices of one dyadic evaluation batch.

    Appendix F of the paper: pick ``kappa`` anchor points uniformly from the evaluation
    set, then build the batch by resampling ``tau`` points from those anchors with
    replacement. The repeated anchors are what make the batch informative about joint
    structure: a model that treats its predictions as independent pays for the same
    mistake ``tau / kappa`` times over.

    Args:
        num_data: size of the evaluation set to draw from
        tau: number of inputs in the batch
        kappa: number of distinct anchor points
        generator: optional random number generator for reproducibility

    Returns:
        indices into the evaluation set, of shape [tau]
    """
    if num_data < 1:
        raise ValueError(f"num_data must be at least 1, got {num_data}.")
    if kappa < 1:
        raise ValueError(f"kappa must be at least 1, got {kappa}.")

    anchors = torch.randint(0, num_data, (kappa,), generator=generator)
    picks = torch.randint(0, kappa, (tau,), generator=generator)
    return anchors[picks]


def joint_log_loss_dyadic(
    logits: Tensor,
    targets: Tensor,
    tau: int = 10,
    kappa: int = 2,
    num_batches: int = 1000,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Average joint log-loss over dyadic batches drawn from an evaluation set.

    The log-loss is the negated joint log-likelihood, normalized by ``tau`` so that
    values are comparable across batch sizes, and averaged over ``num_batches``
    independently drawn dyadic batches.

    Args:
        logits: ENN logits over the whole evaluation set, of shape
            [num_samples, num_data, num_classes]
        targets: class indices of shape [num_data]
        tau: number of inputs per dyadic batch
        kappa: number of distinct anchor points per batch
        num_batches: number of dyadic batches to average over
        generator: optional random number generator for reproducibility

    Returns:
        scalar average joint log-loss per input
    """
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits of shape [num_samples, num_data, num_classes], "
            f"got shape {tuple(logits.shape)}."
        )

    num_data = logits.shape[1]
    total = torch.zeros((), dtype=torch.float64, device=logits.device)
    for _ in range(num_batches):
        idx = dyadic_batch_indices(num_data, tau, kappa, generator).to(logits.device)
        total = total - joint_log_likelihood(logits[:, idx, :], targets[idx])
    return total / (num_batches * tau)


def marginal_log_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Marginal log-loss per input over an evaluation set.

    The ``tau = 1`` counterpart of :func:`joint_log_loss_dyadic`, and the quantity most
    UQ benchmarks report as "NLL".

    Args:
        logits: ENN logits of shape [num_samples, num_data, num_classes]
        targets: class indices of shape [num_data]

    Returns:
        scalar average marginal log-loss per input
    """
    return -marginal_log_likelihood(logits, targets) / logits.shape[1]


def categorical_kl(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """KL divergence between two batches of categorical distributions.

    Args:
        p: reference probabilities of shape [..., num_classes]
        q: comparison probabilities of shape [..., num_classes]
        eps: floor applied to both distributions to keep the logarithms finite

    Returns:
        KL divergence of shape [...], summed over the class axis
    """
    p = p.double().clamp_min(eps)
    q = q.double().clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)
