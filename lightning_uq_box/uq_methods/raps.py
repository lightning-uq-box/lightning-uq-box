"""Regularized Adaptive Prediction Sets (RAPS)."""

"""Adapted from https://github.com/aangelopoulos/conformal_classification"""

import time
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .base import PosthocBase
from .temp_scaling import run_temperature_optimization, temp_scale_logits
from .utils import _get_num_inputs, _get_num_outputs, default_classification_metrics


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: float = None,
        lambda_param: float = None,
        randomized: bool = False,
        allow_zero_sets: bool = False,
        pct_param_tune: float = 0.3,
        lambda_criterion: str = "size",
    ) -> None:
        """Initialize RAPS.

        Args:
            model: model to be calibrated with Temperature S
            optim_lr: learning rate for optimizer
            max_iter: maximum number of iterations to run optimizer
        """
        super().__init__(model)

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optim_lr = optim_lr
        self.max_iter = max_iter
        self.criterion = nn.CrossEntropyLoss()

        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.pct_param_tune = pct_param_tune
        self.num_classes = _get_num_outputs(model)
        self.lambda_criterion = lambda_criterion

        self.kreg = kreg
        self.lambda_param = lambda_param
        if kreg == None or lambda_param == None:
            kreg, lamda, calib_logits = pick_parameters(
                model,
                calib_logits,
                alpha,
                kreg,
                lamda,
                randomized,
                allow_zero_sets,
                pct_param_tune,
                batch_size,
                lambda_criterion,
            )

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step after running posthoc fitting methodology."""
        raise NotImplementedError

    def adjust_model_logits(self, model_logits: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_logits: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model logits tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError

    def on_validation_end(self) -> None:
        """Apply RASP conformal method."""
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        optimizer = partial(torch.optim.LBFGS, lr=self.optim_lr, max_iter=self.max_iter)
        self.temperature = run_temperature_optimization(
            optimizer, self.temperature, all_logits, all_labels, self.criterion
        )
        import pdb

        pdb.set_trace()
        print(0)


def pick_parameters(
    model,
    calib_logits,
    alpha,
    kreg,
    lamda,
    randomized,
    allow_zero_sets,
    pct_paramtune,
    batch_size,
    lambda_criterion,
):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    #  dataloader with only a percentage of the calib logets and the remaining ones
    paramtune_logits, calib_logits = tdata.random_split(
        calib_logits, [num_paramtune, len(calib_logits) - num_paramtune]
    )
    paramtune_loader = tdata.DataLoader(
        paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    calib_loader = tdata.DataLoader(
        calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    if kreg == None:
        # only tune kreg with paramtune data
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lambda_criterion == "size":
            lamda = pick_lamda_size(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
        elif lambda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(
                model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets
            )
    return kreg, lamda, calib_logits


def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array(
        [
            np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0]
            for x in paramtune_logits
        ]
    )
    kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation="higher") + 1
    return kstar


def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [
        0.001,
        0.01,
        0.1,
        0.2,
        0.5,
    ]:  # predefined grid, change if more precision desired.
        # this is somehow running a hole procedure again, but don't know why we need dataloaders for this
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
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(
            paramtune_loader, conformal_model, print_bool=False
        )
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star


def pick_lamda_adaptiveness(
    model,
    paramtune_loader,
    alpha,
    kreg,
    randomized,
    allow_zero_sets,
    strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]],
):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
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


def sort_sum(scores):
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=["size", "correct"])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(
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


def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter("batch_time")
        top1 = AverageMeter("top1")
        top5 = AverageMeter("top5")
        coverage = AverageMeter("RAPS coverage")
        size = AverageMeter("RAPS size")
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            top1.update(prec1.item() / 100.0, n=x.shape[0])
            top5.update(prec5.item() / 100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(
                    f"\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})",
                    end="",
                )
    if print_bool:
        print("")  # Endline

    return top1.avg, top5.avg, coverage.avg, size.avg


class AverageMeter:
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
