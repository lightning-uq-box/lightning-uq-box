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
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
        threshold_model: nn.Module | None = None,
    ) -> None:
        """Initialize AT; configure Trainer(max_epochs=max_epochs) for the first fit.

        The second fit requires a fresh Trainer(max_epochs=1) and independent
        calibration data. The default threshold network may download pretrained weights.

        Args:
            model: fitted base model mapping [batch_size x C x H x W] images to
                binary logits [batch_size x 1 x H x W]
            alpha: target false-negative rate in (0, 1)
            lr: learning rate of the threshold network
            max_epochs: recommended epochs for the first Trainer
            optimizer: optimizer for the threshold network
            lr_scheduler: optional scheduler monitoring ``val_loss``
            save_preds: whether to save test predictions as HDF5 files
            threshold_model: network mapping images [batch_size x C x H x W] and
                probabilities [batch_size x 1 x H x W] to thresholds in [0, 1] of
                shape [batch_size]; ``None`` builds the paper's ImageNet-pretrained
                :class:`~lightning_uq_box.models.ThresholdPredictor`, which may
                download weights. Pass the same kind of network again to
                ``load_from_checkpoint``.

        Raises:
            ValueError: if max_epochs is not positive
        """
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        super().__init__(
            model,
            ThresholdPredictor() if threshold_model is None else threshold_model,
            alpha,
            lr,
            optimizer,
            lr_scheduler,
            save_preds,
        )
        self.save_hyperparameters({"max_epochs": max_epochs})
        self.max_epochs = max_epochs

    def _training_step_network(self, batch: dict[str, Tensor]) -> Tensor:
        """Regress oracle thresholds with plain mean squared error.

        Args:
            batch: images [batch_size x C x H x W] and binary masks

        Returns:
            scalar loss against the oracle thresholds [batch_size]
        """
        X = batch[self.input_key]
        phat = self._probabilities(X)
        oracle_tau = compute_oracle_threshold(phat, batch[self.target_key], self.alpha)
        return F.mse_loss(self._thresholds(X, phat), oracle_tau)
