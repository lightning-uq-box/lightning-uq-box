# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Metrics for uncertainty quantification."""

# TODO eventually these can hopefully be moved to torchmetrics

from typing import List

import torch
from torchmetrics import Metric


class EmpiricalCoverage(Metric):
    """Empirical Coverage."""

    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False):
        """Initialize a new instance of Empirical Coverage Metric.

        Args:
            compute_on_step: Forward only calls update and return None if this is
                set to False
            dist_sync_on_step: Synchronize metric state across processes at each step if
                this is set to True
        """
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

    # TODO make this method accept a list of tensors of just a prediction tensor
    def update(self, pred_set: List[torch.Tensor], targets: torch.Tensor):
        """Update the state with the prediction set and targets.

        Args:
            pred_set: List of tensors of predicted labels for each sample in the batch.
            targets: Tensor of true labels shape [batch_size, num_classes]
        """
        covered = 0
        for i in range(targets.shape[0]):
            if targets[i].item() in pred_set[i]:
                covered += 1
        self.covered += covered
        self.total += targets.shape[0]

    def compute(self) -> float:
        """Compute the coverage of the prediction sets.

        Returns:
            The coverage of the prediction sets.
        """
        return self.covered / self.total
