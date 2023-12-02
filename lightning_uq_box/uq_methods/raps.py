# MIT License
# Copyright (c) 2020 Anastasios Angelopoulos

# Implementation adapted from above
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Regularized Adaptive Prediction Sets (RAPS).

Adapted from https://github.com/aangelopoulos/conformal_classification
"""

from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .base import PosthocBase
from .temp_scaling import run_temperature_optimization, temp_scale_logits
from .utils import default_classification_metrics


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: float = 5,
        lamda_param: float = 0.01,
        randomized: bool = False,
        allow_zero_sets: bool = False,
        pct_param_tune: float = 0.3,
        lamda_criterion: str = "size",
        task: str = "multiclass",
    ) -> None:
        """Initialize RAPS.

        Args:
            model: model to be calibrated with Temperature S
            optim_lr: learning rate for optimizer
            max_iter: maximum number of iterations to run optimizer
            alpha: 1 - alpha is the desired coverage
            kreg: regularization param (smaller kreg leads to smaller sets)
            lamda_param: regularization param (larger lamda leads to smaller sets)
            randomized: whether to use randomized version of conformal prediction
            allow_zero_sets: whether to allow sets of size zero
            pct_param_tune: fraction of calibration data to use for parameter tuning
            lamda_criterion: optimize for 'size' or 'adaptiveness'
            task: task type, one of 'binary', 'multiclass', or 'multilabel'
        """
        super().__init__(model)

        self.num_classes = self.num_outputs
        assert task in self.valid_tasks
        self.task = task
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optim_lr = optim_lr
        self.max_iter = max_iter
        self.loss_fn = nn.CrossEntropyLoss()

        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.pct_param_tune = pct_param_tune
        self.lamda_criterion = lamda_criterion

        self.alpha = alpha

        self.kreg = kreg
        self.lamda_param = lamda_param

        self.penalties = torch.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda_param

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def compute_q_hat(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute q_hat."""
        scores = temp_scale_logits(logits, self.temperature)
        sorted_score_indices, ordered, cumsum = sort_sum(scores)

        E = gen_inverse_quantile_function(
            scores, targets, sorted_score_indices, ordered, cumsum, self.penalties, True, True
        )

        Qhat = torch.quantile(E, 1 - self.alpha, interpolation="higher")

        return Qhat

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology."""
        pred_dict = self.predict_step(batch[self.input_key])

        # logging metrics
        self.log("test_loss", self.loss_fn(pred_dict["pred"], batch[self.target_key]))
        self.test_metrics(pred_dict["pred"], batch[self.target_key])
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict steps via Monte Carlo Sampling.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            logits and conformalized prediction sets
        """
        with torch.no_grad():
            logits = self.model(X)
            scores = temp_scale_logits(logits, self.temperature)
            S = self.adjust_model_logits(logits)

        return {"pred": scores, "pred_set": S}

    def adjust_model_logits(self, model_logits: Tensor) -> Tensor:
        """Adjust model output according to RAPS with fitted Qhat.

        Args:
            model_logits: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model logits tensor of shape [batch_size x num_outputs]
        """
        scores = temp_scale_logits(model_logits, self.temperature)
        sorted_score_indices, ordered, cumsum = sort_sum(scores)

        return gen_cond_quantile_function(
            scores,
            self.Qhat,
            sorted_score_indices,
            ordered,
            cumsum,
            self.penalties,
            self.randomized,
            self.allow_zero_sets,
        )

    def on_validation_end(self) -> None:
        """Apply RASP conformal method."""
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        optimizer = partial(torch.optim.LBFGS, lr=self.optim_lr, max_iter=self.max_iter)
        self.temperature = run_temperature_optimization(
            optimizer, self.temperature, all_logits, all_labels, self.loss_fn
        )

        self.Qhat = self.compute_q_hat(all_logits, all_labels)

        self.post_hoc_fitted = True


def gen_inverse_quantile_function(
    scores, targets, sorted_score_indices, ordered, cumsum, penalties, randomized, allow_zero_sets
) -> Tensor:
    """Generalized inverse quantile conformity score function.

    E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.

    Returns:
        E: generalized inverse quantile conformity score
    """
    E = -torch.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_single_tau(
            targets[i].item(),
            sorted_score_indices[i : i + 1, :],
            ordered[i : i + 1, :],
            cumsum[i : i + 1, :],
            penalties[0, :],
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )

    return E


def gen_cond_quantile_function(
    scores: Tensor, tau: Tensor, sorted_score_indices: Tensor, ordered: Tensor, cumsum: Tensor, penalties: Tensor, randomized: bool, allow_zero_sets: bool
) -> list[Tensor]:
    # TODO why only one tau and not one per batch sample
    """Generalized conditional quantile function.
    
    Args:
        scores: shape [batch_size x num_classes]
        tau: no shape should become Q_hat?
        sorted_score_indices: shape [batch_size x num_classes]
        ordered: shape [batch_size x num_classes]
        cumsum: shape [batch_size x num_classes]
        penalties: shape [1 x num_classes]
        randomized: whether to use randomized version of conformal prediction
        allow_zero_sets: whether to allow sets of size zero
    
    Returns:
        prediction sets
    """
    penalties = penalties.to(scores.device)
    tau = tau.to(scores.device)
  
    penalties_cumsum = torch.cumsum(penalties, dim=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(dim=1).to(scores.device) + 1
    sizes_base = torch.minimum(
        sizes_base, torch.ones_like(sizes_base) * scores.shape[1]
    )

    if randomized:
        V = torch.zeros_like(sizes_base)
        for i in range(sizes_base.shape[0]):
            V[i] = (
                1
                / ordered[i, sizes_base[i] - 1]
                * (
                    tau
                    - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1])
                    - penalties_cumsum[0, sizes_base[i] - 1]
                )
            )  # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (torch.rand(V.shape) >= V).int()
    else:
        sizes = sizes_base

    # always predict max size if alpha==0. (Avoids numerical error.)
    if tau == 1.0:
        sizes[:] = cumsum.shape[1]

    # allow the user the option to never have empty sets (will lead to incorrect
    # coverage if 1-alpha < model's top-1 accuracy
    if not allow_zero_sets:
        sizes[sizes == 0] = 1

    # Construct S from equation (5)
    pred_sets: list[Tensor] = []
    for i in range(sorted_score_indices.shape[0]):
        pred_sets.append(sorted_score_indices[i, 0 : sizes[i]])

    return pred_sets


def get_single_tau(target, sorted_score_indices, ordered, cumsum, penalty, randomized, allow_zero_sets: bool) -> Tensor:
    """Get tau for one example.
    
    Args:
        target:
        sorted_score_indices:
        ordered:
        cumsum:
        penalty:
        randomized:
        allow_zero_sets:

    Returns:
        single tau
    """
    idx = torch.where(sorted_score_indices == target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]

    U = np.random.random()

    if idx == (0, 0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0]
    else:
        return (
            U * ordered[idx]
            + cumsum[(idx[0], idx[1] - 1)]
            + (penalty[0 : (idx[1][0] + 1)]).sum()
        )


def sort_sum(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sort and sum scores.
    
    Args:
        scores:

    Returns:
        sorted_score_indices, ordered, cumsum
    """
    sorted_score_indices = torch.argsort(scores, dim=1, descending=True)
    ordered = torch.sort(scores, dim=1, descending=True).values
    cumsum = torch.cumsum(ordered, dim=1)
    return sorted_score_indices, ordered, cumsum
