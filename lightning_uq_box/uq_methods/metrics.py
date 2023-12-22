# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Metrics for uncertainty quantification."""

# TODO eventually these can hopefully be moved to torchmetrics
from typing import List, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric


class EmpiricalCoverage(Metric):
    """Empirical Coverage."""

    def __init__(self, alpha: float = 0.1, topk: Optional[int] = None, **kwargs):
        """Initialize a new instance of Empirical Coverage Metric.

        Args:
            alpha: 1 - alpha is desired coverage, this is being used if we
                have a prediction tensor, and will choose the set size such
                that coverage is 1-alpha
            topk: if a prediction tensor is used as a prediction set, this is the topk
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.covered = 0
        self.total = 0
        self.set_size = 0
        self.topk = topk

    def update(
        self, pred_set: Union[List[torch.Tensor], Tensor], targets: torch.Tensor
    ) -> None:
        """Update the state with the prediction set and targets.

        Args:
            pred_set: List of tensors of predicted labels for each sample in the batch
                or alternatively a tensor of shape [batch_size, num_samples]
                in which case you take topk to get the predicted labels
            targets: Tensor of true labels shape [batch_size, num_classes]
        """
        covered = 0
        set_size = 0
        if isinstance(pred_set, torch.Tensor):
            if self.topk is not None:
                _, topk_pred_set = pred_set.topk(self.topk, dim=1)
                for i in range(targets.shape[0]):
                    if targets[i].item() in topk_pred_set[i]:
                        covered += 1
                set_size = self.topk * targets.shape[0]
            else:
                for k in range(1, pred_set.shape[1] + 1):
                    _, topk_pred_set = pred_set.topk(k, dim=1)
                    batch_covered = 0
                    for i in range(targets.shape[0]):
                        if targets[i].item() in topk_pred_set[i]:
                            batch_covered += 1
                    coverage = batch_covered / targets.shape[0]
                    if coverage >= 1 - self.alpha:
                        set_size += k * batch_covered
                        covered += batch_covered
                        break
                    covered += batch_covered
        # list of tensors denoting prediction sets
        # fore each sample in the batch
        else:
            for i in range(targets.shape[0]):
                if targets[i].item() in pred_set[i]:
                    covered += 1
                set_size += len(pred_set[i])
        self.covered += covered
        self.set_size += set_size
        self.total += targets.shape[0]

    def compute(self) -> dict[str, float]:
        """Compute the coverage of the prediction sets.

        Returns:
            The coverage of the prediction sets.
        """
        # compute average coverage and set size
        return {
            "coverage": self.covered / self.total,
            "set_size": self.set_size / self.total,
        }
