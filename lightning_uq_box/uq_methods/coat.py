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

        Args:
            model: fitted base model mapping [batch_size x C x H x W] images to
                binary logits [batch_size x 1 x H x W]
            alpha: target false-negative rate in (0, 1)
            lr: learning rate of the threshold network
            max_epochs: recommended epochs for the first Trainer
            temperature: divisor of the soft foreground indicator
            pretrained_threshold_net: whether the threshold ResNet-50 starts from
                ImageNet weights
            optimizer: optimizer for the threshold network
            lr_scheduler: optional scheduler monitoring ``val_loss``
            save_preds: whether to save test predictions as HDF5 files

        Raises:
            ValueError: if max_epochs is not positive
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
        """Optimize squared soft recall gaps without oracle threshold labels.

        Args:
            batch: images [batch_size x C x H x W] and binary masks

        Returns:
            scalar soft miscoverage loss
        """
        X = batch[self.input_key]
        phat = self._probabilities(X)
        return self.soft_miscoverage_loss(
            phat, batch[self.target_key], self.threshold_model(X, phat)
        )
