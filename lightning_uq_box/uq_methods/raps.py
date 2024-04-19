# Apache License 2.0
# Copyright (c) 2020 Anastasios Angelopoulos

# Implementation adapted from above
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Regularized Adaptive Prediction Sets (RAPS).

Adapted from https://github.com/aangelopoulos/conformal_classification
to be integrated with Lightning and port several functions to pytorch.
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchmetrics import Accuracy, CalibrationError, MetricCollection

from .base import PosthocBase
from .metrics import EmpiricalCoverage, SetSize
from .temp_scaling import run_temperature_optimization, temp_scale_logits
from .utils import process_classification_prediction, save_classification_predictions


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    Conformal prediction method for classification tasks, as
    introduced by `Angelopoulos et al. (2020) <https://arxiv.org/abs/2009.14193>`_.

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    valid_tasks = ["binary", "multiclass", "multilable"]
    valid_lambda_criterion = ["size", "adaptiveness"]
    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: LightningModule | nn.Module,
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: int | None = None,
        lamda_param: float | None = None,
        randomized: bool = False,
        allow_zero_sets: bool = False,
        pct_param_tune: float = 0.3,
        lamda_criterion: str = "size",
        task: str = "multiclass",
    ) -> None:
        """Initialize RAPS.

        Args:
            model: model to be calibrated with RAPS
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
        self.temperature = nn.Parameter(torch.Tensor([1.3]))
        self.optim_lr = optim_lr
        self.max_iter = max_iter
        self.loss_fn = nn.CrossEntropyLoss()

        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.pct_param_tune = pct_param_tune

        assert lamda_criterion in self.valid_lambda_criterion, (
            f"lamda_criterion must be one of {self.valid_lambda_criterion}, "
            f"got {lamda_criterion}"
        )
        self.lamda_criterion = lamda_criterion

        self.alpha = alpha

        self.kreg = kreg
        self.lamda_param = lamda_param

        self.setup_task()

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.test_metrics = MetricCollection(
            {
                "Acc": Accuracy(task=self.task, num_classes=self.num_classes),
                "Calibration": CalibrationError(
                    self.task, num_classes=self.num_classes
                ),
                "Empirical Coverage": EmpiricalCoverage(alpha=self.alpha),
                "Set Size": SetSize(alpha=self.alpha),
            },
            prefix="test",
        )

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology."""
        # need to set manually because of inference mode
        self.eval()
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key]

        # logging metrics
        self.log(
            "test_loss",
            self.loss_fn(pred_dict["pred"], batch[self.target_key]),
            batch_size=batch[self.input_key].shape[0],
        )
        self.test_metrics(pred_dict["pred"], batch[self.target_key])
        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict step with RAPS applied.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            logits and conformalized prediction sets
        """
        # need to set manually because of inference mode
        self.eval()
        with torch.no_grad():
            if hasattr(self.model, "predict_step"):
                logits = self.model.predict_step(X)["logits"]
            else:
                logits = self.model(X)
            scores = temp_scale_logits(logits, self.temperature)
            S = self.adjust_model_logits(logits)

        def identity(x, dim=None):
            return x

        pred_dict = process_classification_prediction(scores, aggregate_fn=identity)
        pred_dict["pred_set"] = S
        pred_dict["size"] = torch.tensor([len(x) for x in S])

        return pred_dict

    def adjust_model_logits(self, model_logits: Tensor) -> Tensor:
        """Adjust model output according to RAPS with fitted Qhat.

        Args:
            model_logits: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model logits tensor of shape [batch_size x num_outputs]
        """
        scores = temp_scale_logits(model_logits, self.temperature)
        softmax_scores = F.softmax(scores, dim=1)
        sorted_score_indices, ordered, cumsum = sort_sum(softmax_scores)

        return gen_cond_quantile_function(
            softmax_scores,
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
        self.eval()
        calib_logits = torch.cat(self.model_logits, dim=0).detach()
        calib_labels = torch.cat(self.labels, dim=0).detach()

        calib_ds = TensorDataset(calib_logits, calib_labels)

        # check kreg and lamda param
        if self.kreg is None or self.lamda_param is None:
            num_paramtune = int(np.ceil(self.pct_param_tune * len(calib_logits)))
            paramtune_ds, calib_ds = random_split(
                calib_ds, [num_paramtune, len(calib_logits) - num_paramtune]
            )
            paramtune_loader = DataLoader(paramtune_ds, batch_size=64, shuffle=False)
            if self.kreg is None:
                self.kreg = find_kreg_param(paramtune_loader, self.alpha)

            if self.lamda_param is None:
                if self.lamda_criterion == "size":
                    self.lamda_param = find_lamda_param_size(
                        self.model,
                        self.loss_fn,
                        paramtune_loader,
                        self.alpha,
                        self.kreg,
                        self.randomized,
                        self.allow_zero_sets,
                    )
                elif self.lamda_criterion == "adaptiveness":
                    self.lamda_param = find_lamda_param_adaptiveness(
                        self.model,
                        self.loss_fn,
                        paramtune_loader,
                        self.alpha,
                        self.kreg,
                        self.randomized,
                        self.allow_zero_sets,
                    )

        self.penalties = torch.zeros((1, self.num_classes))
        self.penalties[:, int(self.kreg) :] += self.lamda_param  # noqa: E203

        optimizer = partial(torch.optim.SGD, lr=self.optim_lr)
        self.temperature = run_temperature_optimization(
            calib_logits, calib_labels, self.loss_fn, self.temperature, optimizer
        )

        self.Qhat = compute_q_hat(
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
    softmax_scores = F.softmax(scores, dim=1)
    sorted_score_indices, ordered, cumsum = sort_sum(softmax_scores)

    # TODO why is this always randomized and allow zero sets to true?
    E = gen_inverse_quantile_function(
        targets, sorted_score_indices, ordered, cumsum, penalties, True, True
    )
    Qhat = torch.quantile(E, 1 - alpha, interpolation="higher")
    return Qhat


def find_kreg_param(paramtune_logits: DataLoader, alpha: float) -> int:
    """Find kreg parameter.

    Args:
        paramtune_logits: logits of paramtune data
        alpha: 1 - alpha is the desired coverage

    Returns:
        kreg parameter
    """
    all_ranks = []

    for sample in paramtune_logits:
        # Get the sorted indices of logits in descending order
        sorted_indices = torch.argsort(sample[0], dim=1, descending=True)
        target_expanded = sample[1].unsqueeze(1)
        ranks = (sorted_indices == target_expanded).nonzero(as_tuple=True)[1]
        all_ranks.append(ranks)

    all_ranks = torch.cat(all_ranks)
    kstar = torch.quantile(all_ranks.float(), 1 - alpha, interpolation="higher") + 1

    return int(kstar.cpu().item())


def find_lamda_param_size(
    model: nn.Module,
    loss_fn: nn.Module,
    paramtune_loader: DataLoader,
    alpha: float,
    kreg: float,
    randomized: bool,
    allow_zero_sets: bool,
) -> Tensor:
    """Find lamda parameter for size criterion.

    Args:
        model: model to be used as conformal model
        loss_fn: loss function
        paramtune_loader: paramtune data loader
        alpha: 1 - alpha is the desired coverage
        kreg: regularization param (smaller kreg leads to smaller sets)
        randomized: whether to use randomized version of conformal prediction
        allow_zero_sets: whether to allow sets of size zero
    """
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    lamda_star = 0
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            loss_fn=loss_fn,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )
        size_metric = SetSize(topk=None)
        for i, (logit, target) in enumerate(paramtune_loader):
            _, S = conformal_model(logit)
            size_metric.update(S, target)

        size = size_metric.compute()
        if size < best_size:
            best_size = size
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
    """Find lamda parameter for adaptiveness criterion.

    Args:
        model: model to be used as conformal model
        loss_fn: loss function
        paramtune_loader: paramtune data loader
        alpha: 1 - alpha is the desired coverage
        kreg: regularization param (smaller kreg leads to smaller sets)
        randomized: whether to use randomized version of conformal prediction
        allow_zero_sets: whether to allow sets of size zero
        strata: strata to use for adaptiveness criterion
    """
    lamda_star = 0
    best_violation = 1
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]:
        conformal_model = ConformalModelLogits(
            model,
            paramtune_loader,
            loss_fn=loss_fn,
            alpha=alpha,
            kreg=kreg,
            lamda=temp_lam,
            randomized=randomized,
            allow_zero_sets=allow_zero_sets,
        )
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)

        if curr_violation < best_violation:
            best_violation = curr_violation
            lamda_star = temp_lam
    return lamda_star


def get_violation(
    cmodel: nn.Module,
    loader_paramtune: DataLoader,
    strata: list[list[int]],
    alpha: float,
):
    """Get violation for adaptiveness criterion.

    Args:
        cmodel: conformal model
        loader_paramtune: paramtune data loader
        strata: strata to use for adaptiveness criterion
        alpha: 1 - alpha is the desired coverage

    Returns:
        violation
    """
    df = pd.DataFrame(columns=["size", "correct"])
    for logit, target in loader_paramtune:
        _, S = cmodel(logit)
        size = np.array([x.size()[0] for x in S])
        sorted_score_indices, _, _ = sort_sum(logit)
        sorted_score_indices = sorted_score_indices.cpu().numpy()
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int(target[j] in list(S[j]))
        batch_df = pd.DataFrame({"size": size, "correct": correct})
        df = pd.concat([df, batch_df], ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[(df["size"] >= stratum[0]) & (df["size"] <= stratum[1])]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation


class ConformalModelLogits(nn.Module):
    """Conformal Model for logits."""

    def __init__(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        loss_fn: nn.Module,
        alpha: float,
        kreg: int | None = None,
        lamda: float | None = None,
        randomized: bool = False,
        allow_zero_sets: bool = False,
    ):
        """Initialize ConformalModelLogits.

        Args:
            model: model to be used as conformal model
            calib_loader: calibration data loader
            loss_fn: loss function
            alpha: 1 - alpha is the desired coverage
            kreg: regularization param (smaller kreg leads to smaller sets)
            lamda: regularization param (larger lamda leads to smaller sets)
                (any value of kreg and lambda will lead to coverage, but will yield
                different set sizes)
            randomized: whether to use randomized version of conformal prediction
            allow_zero_sets: whether to allow sets of size zero
            pct_paramtune: fraction of calibration data to use for parameter tuning
            batch_size: batch size for parameter tuning
            lamda_criterion: optimize for 'size' or 'adaptiveness'
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets

        # temperature optimization
        if isinstance(calib_loader.dataset, TensorDataset):
            logits = calib_loader.dataset.tensors[0]
            labels = calib_loader.dataset.tesnors[1]
        elif isinstance(calib_loader.dataset, Subset):
            logits = calib_loader.dataset.dataset.tensors[0]
            labels = calib_loader.dataset.dataset.tensors[1]

        optimizer = partial(torch.optim.SGD, lr=0.01)
        self.temperature = run_temperature_optimization(
            logits,
            labels,
            nn.CrossEntropyLoss(),
            nn.Parameter(torch.Tensor([1.3]).to(logits.device)),
            optimizer,
        )

        self.penalties = torch.zeros((1, calib_loader.dataset[0][0].shape[0]))

        self.penalties[:, kreg:] += lamda
        self.Qhat = compute_q_hat(
            logits, labels, self.temperature, self.penalties, self.alpha
        )

    def forward(
        self,
        logits: Tensor,
        randomized: bool | None = None,
        allow_zero_sets: bool | None = None,
    ):
        """Forward pass.

        Args:
            logits: model output logits of shape [batch_size x num_outputs]
            randomized: whether to use randomized version of conformal prediction
            allow_zero_sets: whether to allow sets of size zero

        Returns:
            logits and prediction sets
        """
        if randomized is None:
            randomized = self.randomized
        if allow_zero_sets is None:
            allow_zero_sets = self.allow_zero_sets

        with torch.no_grad():
            # logits_numpy = logits.detach().cpu().numpy()
            scores = F.softmax(logits / self.temperature.item(), dim=1)
            sorted_score_indices, ordered, cumsum = sort_sum(scores)

            S = gen_cond_quantile_function(
                scores,
                self.Qhat,
                sorted_score_indices=sorted_score_indices,
                ordered=ordered,
                cumsum=cumsum,
                penalties=self.penalties,
                randomized=randomized,
                allow_zero_sets=allow_zero_sets,
            )

        return logits, S


def gen_inverse_quantile_function(
    targets: Tensor,
    sorted_score_indices: Tensor,
    ordered: Tensor,
    cumsum: Tensor,
    penalties: Tensor,
    randomized: bool,
    allow_zero_sets: bool,
) -> Tensor:
    """Generalized inverse quantile conformity score function.

    E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1]
    such that the correct label enters.

    Args:a
        targets: shape [batch_size]
        sorted_score_indices: shape [batch_size x num_classes]
        ordered: shape [batch_size x num_classes]
        cumsum: shape [batch_size x num_classes]
        penalties: shape [1 x num_classes]
        randomized: whether to use randomized version of conformal prediction
        allow_zero_sets: whether to allow sets of size zero

    Returns:
        E: generalized inverse quantile conformity score
    """
    E = -torch.ones((targets.shape[0],))
    for i in range(targets.shape[0]):
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
    target: Tensor,
    sorted_score_indices: Tensor,
    ordered: Tensor,
    cumsum: Tensor,
    penalty: Tensor,
    randomized: bool,
    allow_zero_sets: bool,
) -> Tensor:
    """Get tau for one example.

    Args:
        target: target label
        sorted_score_indices: shape [batch_size x num_classes]
        ordered: shape [batch_size x num_classes]
        cumsum: shape [batch_size x num_classes]
        penalty: shape [1 x num_classes]
        randomized: whether to use randomized version of conformal prediction
        allow_zero_sets: whether to allow sets of size zero

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
    # https://stackoverflow.com/questions/66767610/difference-between-numpy-argsort-and-torch-argsort
    # scores = scores.double() if scores.dtype != torch.float64 else scores
    # sorted_score_indices = torch.argsort(scores, dim=1, descending=True)
    device = scores.device
    sorted_score_indices = scores.cpu().numpy().argsort(axis=1)[:, ::-1]
    sorted_score_indices = torch.from_numpy(sorted_score_indices.copy()).to(device)
    ordered = torch.sort(scores, dim=1, descending=True).values
    cumsum = torch.cumsum(ordered, dim=1)
    return sorted_score_indices, ordered, cumsum
