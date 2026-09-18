# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Conditional Optimization for Adaptive Thresholding in binary segmentation."""

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn

from lightning_uq_box.models import ThresholdPredictor

from .loss_functions import SoftMiscoverageLoss
from .segmentation_conformal_base import SegmentationPosthocBase


class COAT(SegmentationPosthocBase):
    """Optimize differentiable miscoverage, then calibrate marginal false-negative risk."""

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        lr: float = 5e-4,
        max_epochs: int = 60,
        temperature: float = 0.05,
        pretrained_threshold_net: bool = True,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize COAT with temperature as the sigmoid divisor.

        Configure the first Trainer with max_epochs (recommended 60) explicitly;
        use a fresh Trainer(max_epochs=1) for independent calibration data.
        """
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        loss = SoftMiscoverageLoss(temperature, 1 - alpha)
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
                "temperature": temperature,
                "pretrained_threshold_net": pretrained_threshold_net,
            }
        )
        self.max_epochs = max_epochs
        self.soft_miscoverage_loss = loss

    def _training_step_network(self, batch: dict[str, Tensor]) -> Tensor:
        """Optimize squared soft recall gaps without oracle threshold labels."""
        X = batch[self.input_key]
        phat = self._probabilities(X)
        return self.soft_miscoverage_loss(
            phat, batch[self.target_key], self.threshold_model(X, phat)
        )
