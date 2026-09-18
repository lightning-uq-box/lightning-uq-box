# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Supervised adaptive thresholding for binary conformal segmentation."""

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn
from torch.nn import functional as F

from lightning_uq_box.models import ThresholdPredictor

from .segmentation_conformal_base import SegmentationPosthocBase
from .segmentation_conformal_utils import compute_oracle_threshold


class AdaptiveThresholding(SegmentationPosthocBase):
    """Regress conservative per-image oracle thresholds, then calibrate marginal FNR."""

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        lr: float = 1e-4,
        max_epochs: int = 30,
        pretrained_threshold_net: bool = True,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize AT; configure Trainer(max_epochs=max_epochs) for the first fit.

        The second fit requires a fresh Trainer(max_epochs=1) and independent
        calibration data. Pretrained weights may download on first construction.
        """
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        super().__init__(
            model,
            ThresholdPredictor(pretrained_threshold_net),
            alpha,
            lr,
            optimizer,
            lr_scheduler,
            save_preds,
        )
        self.save_hyperparameters(
            {
                "max_epochs": max_epochs,
                "pretrained_threshold_net": pretrained_threshold_net,
            }
        )
        self.max_epochs = max_epochs

    def _training_step_network(self, batch: dict[str, Tensor]) -> Tensor:
        """Regress oracle thresholds [B] with plain mean squared error."""
        X = batch[self.input_key]
        phat = self._probabilities(X)
        oracle_tau = compute_oracle_threshold(phat, batch[self.target_key], self.alpha)
        return F.mse_loss(self.threshold_model(X, phat), oracle_tau)
