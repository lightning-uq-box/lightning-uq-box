# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Two-stage training and calibration for binary adaptive segmentation."""

import os
from typing import Any

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from .base import PosthocBase
from .metrics import CoverageGap, PerImageCoverage
from .segmentation_conformal_utils import (
    binary_segmentation_inputs,
    find_calibration_shift,
)
from .utils import save_image_predictions


class SegmentationPosthocBase(PosthocBase):
    """Train a threshold network, then calibrate on an independent binary-mask split.

    The base segmentation model must be already fitted and emit logits [B,1,H,W].
    First call Trainer.fit on threshold-training data, then use a fresh Trainer
    with max_epochs=1 on calibration data. Final evaluation uses a fourth held-out
    split. Only marginal FNR control is claimed, under exchangeability. Calibration
    currently requires a single process. Stage and shift survive checkpoints.
    """

    pred_dir_name = "preds"
    t_prime: Tensor

    def __init__(
        self,
        model: nn.Module,
        threshold_model: nn.Module,
        alpha: float = 0.1,
        lr: float = 1e-4,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize a frozen base model and a trainable threshold predictor."""
        if not 0 < alpha < 1 or lr <= 0:
            raise ValueError("alpha must lie in (0,1) and lr must be positive.")
        super().__init__(model)
        self.save_hyperparameters({"alpha": alpha, "lr": lr, "save_preds": save_preds})
        self.threshold_model = threshold_model
        self.alpha = alpha
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_preds = save_preds
        self._network_trained = False
        self.register_buffer("t_prime", torch.tensor(0.0))
        self.test_metrics = MetricCollection(
            {
                "coverage": PerImageCoverage(),
                "coverage_gap": CoverageGap(1 - alpha),
                "F1Score": BinaryF1Score(),
                "JaccardIndex": BinaryJaccardIndex(),
            },
            prefix="test_",
            compute_groups=False,
        )
        self._phat: list[Tensor] = []
        self._tau: list[Tensor] = []
        self._labels: list[Tensor] = []

    def train(self, mode: bool = True) -> "SegmentationPosthocBase":
        """Keep the fitted base model in eval mode, including its batch statistics."""
        super().train(mode)
        self.model.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the training/calibration stage."""
        checkpoint["segmentation_conformal_stage"] = (
            self._network_trained,
            self.post_hoc_fitted,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the training/calibration stage."""
        self._network_trained, self.post_hoc_fitted = checkpoint[
            "segmentation_conformal_stage"
        ]

    def on_train_start(self) -> None:
        """Initialize stage-specific state without disabling threshold-network training."""
        self.model.eval()
        if self._network_trained:
            if self.trainer.world_size != 1:
                raise RuntimeError("Calibration requires a single process.")
            self.eval()
            self.post_hoc_fitted = False
            self._phat, self._tau, self._labels = [], [], []

    def _probabilities(self, X: Tensor) -> Tensor:
        """Return frozen foreground probabilities [B,1,H,W]."""
        self.model.eval()
        with torch.no_grad():
            phat = self.model(X).sigmoid()
        if phat.ndim != 4 or phat.shape[1] != 1:
            raise ValueError("Base model must emit binary logits [B,1,H,W].")
        return phat

    def _training_step_network(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the subclass-specific threshold-network loss."""
        raise NotImplementedError

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor | None:
        """Optimize thresholds in stage one or gather calibration data in stage two."""
        if not self._network_trained:
            loss = self._training_step_network(batch)
            self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
            return loss
        if self.trainer.current_epoch > 0:
            raise RuntimeError(
                "Post-Hoc methods only need one pass over the calibration data loader."
            )
        with torch.no_grad():
            phat, labels = binary_segmentation_inputs(
                self._probabilities(batch[self.input_key]), batch[self.target_key]
            )
            self._phat.append(phat.cpu())
            self._labels.append(labels.cpu())
            self._tau.append(self.threshold_model(batch[self.input_key], phat).cpu())
        return None

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor | None:
        """Monitor network loss before calibration; calibration never uses validation data."""
        if not self._network_trained:
            loss = self._training_step_network(batch)
            self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
            return loss
        return None

    def on_train_end(self) -> None:
        """Calibrate only after the network-training stage has completed."""
        if self._network_trained:
            if not self._phat:
                raise RuntimeError("Calibration loader was empty.")
            shift = find_calibration_shift(
                torch.cat(self._tau),
                torch.cat(self._phat),
                torch.cat(self._labels),
                1 - self.alpha,
            )
            self.t_prime.fill_(shift)
            self.post_hoc_fitted = True
            self._phat, self._tau, self._labels = [], [], []

    def on_fit_end(self) -> None:
        """Mark the threshold network trained for the next fit call."""
        self._network_trained = True

    def configure_optimizers(self) -> Any:
        """Optimize only the threshold network, or skip optimization during calibration."""
        if self._network_trained:
            return None
        optimizer = self.optimizer(self.threshold_model.parameters(), lr=self.lr)
        if self.lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler(optimizer),
                "monitor": "val_loss",
            },
        }

    def adjust_model_logits(
        self, model_output: tuple[Tensor, Tensor]
    ) -> dict[str, Tensor]:
        """Apply the subtraction shift to probabilities [B,1,H,W] and thresholds [B]."""
        phat, predicted_tau = model_output
        tau = (predicted_tau - self.t_prime).clamp(0, 1)
        return {
            "pred": phat >= tau[:, None, None, None],
            "phat": phat,
            "tau": tau,
            "pred_uct": tau,
        }

    @torch.no_grad()
    def forward(self, X: Tensor) -> dict[str, Tensor]:
        """Return calibrated masks [B,1,H,W], probabilities and thresholds [B]."""
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted; train the threshold network, then fit on calibration data."
            )
        phat = self._probabilities(X)
        return self.adjust_model_logits((phat, self.threshold_model(X, phat)))

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict calibrated binary masks for images [B,3,H,W]."""
        return self(X)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Evaluate masks and per-image coverage on held-out data."""
        out = self.predict_step(batch[self.input_key])
        _, labels = binary_segmentation_inputs(out["phat"], batch[self.target_key])
        self.test_metrics.update(out["pred"], labels)
        out[self.target_key] = labels
        return self.add_aux_data_to_dict(out, batch)

    def on_test_start(self) -> None:
        """Create the prediction directory when saving is enabled."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if self.save_preds:
            os.makedirs(self.pred_dir, exist_ok=True)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Save per-image predictions; scalar thresholds are HDF5 attributes."""
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)
