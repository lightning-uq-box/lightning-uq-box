# MIT License
# Copyright (c) 2020 Anastasios Angelopoulos

# Implementation adapted from above
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Regularized Adaptive Prediction Sets (RAPS).

Adapted from https://github.com/aangelopoulos/conformal_classification
"""

import os
from functools import partial
from typing import Optional, Union
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
import pandas as pd

from .base import PosthocBase
from .temp_scaling import run_temperature_optimization, temp_scale_logits
from .utils import default_classification_metrics, save_classification_predictions


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    valid_tasks = ["binary", "multiclass", "multilable"]
    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: Optional[int] = None,
        lamda_param: Optional[float] = None,
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
                (any value of kreg and lambda will lead to coverage, but will yield
                different set sizes)
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

        self.setup_task()

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology."""
        # need to set manually because of inference mode
        self.eval()
        pred_dict = self.predict_step(batch[self.input_key])

        # logging metrics
        self.log("test_loss", self.loss_fn(pred_dict["pred"], batch[self.target_key]))
        self.test_metrics(pred_dict["pred"], batch[self.target_key])
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with RAPS applied.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            logits and conformalized prediction sets
        """
        # need to set manually because of inference mode
        self.eval()
        with torch.no_grad():
            # TODO (nils): check if the underlying model is a lightning module
            # and has a predict step, because then RAPs should be applied on top of that
            if hasattr(self.model, "predict_step"):
                logits = self.model.predict_step(X)["logits"]
            else:
                logits = self.model(X)
            scores = temp_scale_logits(logits, self.temperature)
            S = self.adjust_model_logits(logits)

        return {"pred": scores, "pred_set": S, "logits": scores}

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
        calib_logits = torch.cat(self.model_logits, dim=0).detach()
        calib_labels = torch.cat(self.labels, dim=0).detach()

        # check kreg and lamda param
        if self.kreg is None or self.lamda_param is None:
            num_paramtune = int(np.ceil(self.pct_paramtune * len(calib_logits)))
            paramtune_logits, calib_logits = random_split(
                calib_logits, [num_paramtune, len(calib_logits) - num_paramtune]
            )
            paramtune_loader = DataLoader(
                paramtune_logits, batch_size=64, shuffle=False, pin_memory=True
            )
            if self.kreg is None:
                self.kreg = find_kreg_param(paramtune_loader, self.alpha)

            if self.lamda_param is None:
                if self.lamda_criterion == "size":
                    self.lamda_param = self.find_lamda_param_size(
                        self.model,
                        paramtune_loader,
                        self.alpha,
                        self.kreg,
                        self.randomized,
                        self.allow_zero_sets,
                    )
                elif self.lamda_criterion == "adaptiveness":
                    self.lamda_param = self.find_lamda_param_adaptiveness(
                        calib_logits, calib_labels, self.kreg
                    )

        self.penalties = torch.zeros((1, self.num_classes))
        self.penalties[:, int(self.kreg) :] += self.lamda_param  # noqa: E203

        optimizer = partial(torch.optim.LBFGS, lr=self.optim_lr, max_iter=self.max_iter)
        self.temperature = run_temperature_optimization(
            optimizer, self.temperature, calib_logits, calib_labels, self.loss_fn
        )

        self.Qhat = self.compute_q_hat(
            calib_logits, calib_labels, self.temperature, self.penalties, self.alpha
        )

        self.post_hoc_fitted = True

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


def compute_q_hat(
    logits: Tensor,
    targets: Tensor,
    temperature: nn.Parameter,
    penalties: Tensor,
    alpha: float,
) -> Tensor:
    """Compute q_hat.

    Args:
        logits: model output logits of shape [batch_size x num_outputs]
        targets: labels of shape [batch_size]
        temperature: temperature parameter
        penalties: regularization penalties
        alpha: 1 - alpha is the desired coverage

    Returns:
        q_hat
    """
    scores = temp_scale_logits(logits, temperature)
    sorted_score_indices, ordered, cumsum = sort_sum(scores)

    E = gen_inverse_quantile_function(
        scores, targets, sorted_score_indices, ordered, cumsum, penalties, True, True
    )

    Qhat = torch.quantile(E, 1 - alpha, interpolation="higher")

    return Qhat


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]


def find_kreg_param(paramtune_logits: DataLoader, alpha: float):
    """Find kreg parameter."""
    gt_locs_kstar = torch.tensor(
        [
            torch.where(torch.argsort(x[0], descending=True) == x[1])[0][0]
            for x in paramtune_logits
        ]
    )
    kstar = torch.quantile(gt_locs_kstar.float(), 1 - alpha, interpolation="higher") + 1
    return kstar


def find_lamda_param_size(
    model: nn.Module,
    loss_fn: nn.Module,
    paramtune_loader: DataLoader,
    alpha: float,
    kreg: float,
    randomized: bool,
    allow_zero_sets: bool,
) -> Tensor:
    """Find lamda parameter for size criterion."""

    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [
        0.001,
        0.01,
        0.1,
        0.2,
        0.5,
    ]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            loss_fn=loss_fn,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
        )
        coverage = AverageMeter("RAPS coverage")
        size = AverageMeter("RAPS size")
        for i, (x, target) in enumerate(paramtune_loader):
            target = target.cuda()
            # compute output
            _, S = model(x.cuda())

            cvg, sz = coverage_size(S, target)
            # Update meters
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

        if size.avg < best_size:
            best_size = size.avg
            lamda_star = temp_lam

    return lamda_star


def find_lamda_param_adaptiveness(
    model: nn.Module,
    loss_fn: nn.Module,
    paramtune_loader: DataLoader,
    alpha: float,
    kreg: float,
    randomized: bool,
    allow_zero_sets: bool,
    strata: list[list[int]] = [[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]],
) -> Tensor:
    lamda_star = 0
    best_violation = 1
    for temp_lam in [
        0,
        1e-5,
        1e-4,
        8e-4,
        9e-4,
        1e-3,
        1.5e-3,
        2e-3,
    ]:  # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
            naive=False,
        )
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)

        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=["size", "correct"])
    for logit, target in loader_paramtune:
        # compute output
        _, S = cmodel(
            logit
        )  # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int(target[j] in list(S[j]))
        batch_df = pd.DataFrame({"size": size, "correct": correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df["size"] >= stratum[0]) & (df["size"] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation  # the violation


class ConformalModelLogits(nn.Module):
    def __init__(
        self,
        model,
        calib_loader,
        loss_fn: nn.Module,
        alpha,
        kreg=None,
        lamda=None,
        randomized=True,
        allow_zero_sets=False,
        naive=False,
        pct_paramtune=0.3,
        batch_size=32,
        lamda_criterion="size",
    ):
        super(ConformalModelLogits, self).__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets

        # temperature optimization
        optimizer = partial(torch.optim.LBFGS, lr=self.optim_lr, max_iter=self.max_iter)
        logits = calib_loader.dataset[0]
        targets = calib_loader.dataset[1]

        self.T = run_temperature_optimization(
            optimizer=optimizer, logits=logits, targets=targets, criterion=loss_fn
        )

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))

        if not (kreg == None) and not naive:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1 - alpha
        if not naive:
            self.Qhat = compute_q_hat(
                logits, targets, self.T, self.penalties, self.alpha
            )

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        """Forward pass."""
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = F.softmax(logits_numpy / self.T.item(), axis=1)
            I, ordered, cumsum = sort_sum(scores)

            S = gen_cond_quantile_function(
                scores,
                self.Qhat,
                I=I,
                ordered=ordered,
                cumsum=cumsum,
                penalties=self.penalties,
                randomized=randomized,
                allow_zero_sets=allow_zero_sets,
            )

        return logits, S


def gen_inverse_quantile_function(
    scores,
    targets,
    sorted_score_indices,
    ordered,
    cumsum,
    penalties,
    randomized,
    allow_zero_sets,
) -> Tensor:
    """Generalized inverse quantile conformity score function.

    E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1]
    such that the correct label enters.

    Returns:
        E: generalized inverse quantile conformity score
    """
    E = -torch.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_single_tau(
            targets[i].item(),
            sorted_score_indices[i : i + 1, :],  # noqa: E203
            ordered[i : i + 1, :],  # noqa: E203
            cumsum[i : i + 1, :],  # noqa: E203
            penalties[0, :],  # noqa: E203
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )

    return E


def gen_cond_quantile_function(
    scores: Tensor,
    tau: Tensor,
    sorted_score_indices: Tensor,
    ordered: Tensor,
    cumsum: Tensor,
    penalties: Tensor,
    randomized: bool,
    allow_zero_sets: bool,
) -> list[Tensor]:
    """Generalized conditional quantile function.

    Args:
        scores: shape [batch_size x num_classes]
        tau: no shape, uses Q_hat
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
            sizes = sizes_base - (torch.rand(V.shape).to(scores.device) >= V).int()
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
        pred_sets.append(sorted_score_indices[i, 0 : sizes[i]])  # noqa: E203

    return pred_sets


def get_single_tau(
    target,
    sorted_score_indices,
    ordered,
    cumsum,
    penalty,
    randomized,
    allow_zero_sets: bool,
) -> Tensor:
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
            + (penalty[0 : (idx[1][0] + 1)]).sum()  # noqa: E203
        )


def sort_sum(scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sort and sum scores.

    Args:
        scores: model scores [batch_size x num_classes]

    Returns:
        sorted_score_indices, ordered, cumsum
    """
    sorted_score_indices = torch.argsort(scores, dim=1, descending=True)
    ordered = torch.sort(scores, dim=1, descending=True).values
    cumsum = torch.cumsum(ordered, dim=1)
    return sorted_score_indices, ordered, cumsum
