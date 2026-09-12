# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Hierarchical Probabilistic U-Net with GECO or fixed-beta ELBO training."""

import math
import os
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.distributions import Independent, kl_divergence

from ..models.hierarchical_prob_unet import (
    HierarchicalProbUNet as HierarchicalProbUNetModel,
)
from .base import BaseModule
from .utils import (
    default_segmentation_metrics,
    process_segmentation_prediction,
    save_image_predictions,
)


def hierarchical_ce_loss(
    logits: Tensor,
    target: Tensor,
    top_k_percentage: float | None = 0.02,
    deterministic_top_k: bool = False,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute soft-label CE, optionally mining hard pixels over the whole batch.

    Args:
        logits: Logits [B, C, H, W]; C=1 uses binary cross entropy.
        target: Soft or one-hot targets of the same shape as logits.
        top_k_percentage: Fraction of pixels selected, or None for all pixels.
        deterministic_top_k: Rank CE directly instead of adding Gumbel noise.
        mask: Optional binary valid-pixel mask [B, H, W] or [B, H*W].

    Returns:
        Batch-averaged pixel sum, selected-pixel mean and [B, H*W] mask.
        At least one valid pixel is selected when rounding k would give zero.
    """
    if logits.ndim != 4 or logits.shape != target.shape:
        raise ValueError("logits and target must have matching [B, C, H, W] shapes.")
    if top_k_percentage is not None and not 0 < top_k_percentage <= 1:
        raise ValueError("top_k_percentage must be in (0, 1] or None.")
    if logits.shape[1] == 1:
        xe = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        ).squeeze(1)
    else:
        xe = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    xe = xe.flatten(1)
    valid = (
        torch.ones_like(xe, dtype=torch.bool)
        if mask is None
        else mask.reshape_as(xe).bool()
    )
    selected = valid
    if top_k_percentage is not None:
        k = min(
            max(1, math.floor(xe.numel() * top_k_percentage)), int(valid.sum().item())
        )
        scores = xe.detach().flatten()
        # Normalizing before log only adds a constant and leaves the ranking unchanged.
        scores = scores.clamp_min(torch.finfo(scores.dtype).tiny).log()
        if not deterministic_top_k:
            u = torch.rand_like(scores).clamp_(
                torch.finfo(scores.dtype).eps, 1 - torch.finfo(scores.dtype).eps
            )
            scores = scores - (-u.log()).log()
        scores = scores.masked_fill(~valid.flatten(), -torch.inf)
        selected = torch.zeros_like(valid).flatten()
        selected.scatter_(0, scores.topk(k).indices, True)
        selected = selected.reshape_as(xe)
    weights = selected.to(xe.dtype)
    total = (xe * weights).sum()
    return {
        "sum": total / logits.shape[0],
        "mean": total / weights.sum().clamp_min(1),
        "mask": weights,
    }


class HierarchicalProbUNet(BaseModule):
    """Train an SMP spatial latent hierarchy with GECO (default) or ELBO.

    Binary tasks use one sigmoid logit; multiclass tasks use at least two logits.
    GECO state is checkpointed and updated only in training_step. Persistent
    multiplier saturation at 1e5 usually means kappa is too low for the data.
    """

    valid_tasks: ClassVar[list[str]] = ["multiclass", "binary"]
    valid_loss_types: ClassVar[list[str]] = ["geco", "elbo"]
    pred_dir_name = "preds"
    log_lagmul: Tensor
    ma_rec_loss: Tensor

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss_type: str = "geco",
        kappa: float = 0.05,
        decay: float = 0.99,
        rate: float = 1e-2,
        lagrange_init: float = 1.0,
        beta: float = 1.0,
        top_k_percentage: float | None = 0.02,
        deterministic_top_k: bool = False,
        num_samples: int = 16,
        task: str = "multiclass",
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the model, segmentation metrics and checkpointed GECO state.

        Args:
            model: Hierarchical model exposing sample and reconstruct.
            num_classes: Logit channels, explicitly specified without introspection.
            loss_type: "geco" or "elbo".
            kappa: Target CE per selected valid pixel.
            decay: Reconstruction exponential moving-average decay.
            rate: Multiplicative GECO update rate in log space.
            lagrange_init: Positive initial Lagrange multiplier.
            beta: KL weight for ELBO.
            top_k_percentage: Hard-pixel fraction or None to use all pixels.
            deterministic_top_k: Disable Gumbel perturbation in pixel selection.
            num_samples: Prior samples per prediction.
            task: "multiclass" or single-logit "binary".
            freeze_backbone: Freeze both SMP encoders.
            freeze_decoder: Freeze the deterministic stitching tail and head.
            optimizer: Optimizer factory for network weights.
            lr_scheduler: Optional scheduler factory (epoch interval).
            save_preds: Save test predictions as HDF5 files.
        """
        super().__init__()
        if not callable(getattr(model, "sample", None)) or not callable(
            getattr(model, "reconstruct", None)
        ):
            raise TypeError("model must expose sample() and reconstruct().")
        if task not in self.valid_tasks or loss_type not in self.valid_loss_types:
            raise ValueError("Unsupported task or loss_type.")
        if (task == "binary" and num_classes != 1) or (
            task == "multiclass" and num_classes < 2
        ):
            raise ValueError(
                "binary requires num_classes=1; multiclass requires at least 2."
            )
        if not (
            0 <= decay < 1
            and rate >= 0
            and kappa >= 0
            and beta >= 0
            and 0 < lagrange_init <= 1e5
            and num_samples >= 1
        ):
            raise ValueError("Invalid GECO, beta or sampling hyperparameters.")
        if top_k_percentage is not None and not 0 < top_k_percentage <= 1:
            raise ValueError("top_k_percentage must be in (0, 1] or None.")
        if getattr(model, "classes", num_classes) != num_classes:
            raise ValueError("num_classes must match the model's logit channels.")
        self.save_hyperparameters(ignore=["model", "optimizer", "lr_scheduler"])
        self.model = cast(HierarchicalProbUNetModel, model)
        self.num_classes, self.task = num_classes, task
        self.loss_type, self.kappa, self.decay, self.rate = (
            loss_type,
            kappa,
            decay,
            rate,
        )
        self.beta, self.num_samples = beta, num_samples
        self.top_k_percentage, self.deterministic_top_k = (
            top_k_percentage,
            deterministic_top_k,
        )
        self.freeze_backbone, self.freeze_decoder = freeze_backbone, freeze_decoder
        self.optimizer, self.lr_scheduler, self.save_preds = (
            optimizer,
            lr_scheduler,
            save_preds,
        )
        self.register_buffer("log_lagmul", torch.tensor(math.log(lagrange_init)))
        self.register_buffer("ma_rec_loss", torch.tensor(float("nan")))
        self.train_metrics = default_segmentation_metrics(
            prefix="train", num_classes=num_classes, task=task
        )
        self.val_metrics = default_segmentation_metrics(
            prefix="val", num_classes=num_classes, task=task
        )
        self.test_metrics = default_segmentation_metrics(
            prefix="test", num_classes=num_classes, task=task
        )
        self.freeze_model()

    def freeze_model(self) -> None:
        """Freeze both backbones or only the deterministic stitching tail and head."""
        if self.freeze_backbone:
            self.model.prior_encoder.requires_grad_(False)
            self.model.posterior_encoder.requires_grad_(False)
        if self.freeze_decoder:
            n = len(self.model.latent_dims)
            self.model.prior_decoder.blocks[n:].requires_grad_(False)
            self.model.segmentation_head.requires_grad_(False)

    def _targets(self, target: Tensor) -> Tensor:
        if target.ndim == 4 and target.shape[1] == self.num_classes:
            return target.float()
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if self.task == "binary":
            return target.unsqueeze(1).float()
        return F.one_hot(target.long(), self.num_classes).movedim(-1, 1).float()

    def _geco_constraint(
        self, rec: Tensor, valid_pixels: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Global statistics keep GECO buffers identical across DDP ranks.
        stats = torch.stack([rec.detach(), valid_pixels.detach()])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
            stats /= torch.distributed.get_world_size()
        ma = torch.where(
            self.ma_rec_loss.isnan(),
            stats[0],
            self.decay * self.ma_rec_loss + (1 - self.decay) * stats[0],
        )
        # Forward uses the EMA; backward retains the full current CE gradient.
        constraint = rec + (ma - rec).detach() - self.kappa * stats[1]
        return constraint, ma

    @torch.no_grad()
    def _update_geco(self, constraint: Tensor, ma: Tensor) -> None:
        self.ma_rec_loss.copy_(ma)
        self.log_lagmul.add_(self.rate * constraint.detach()).clamp_(
            math.log(1e-5), math.log(1e5)
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return loss, per-scale KL nats and [B, C, H, W] reconstruction logits."""
        logits, q, p = self.reconstruct(
            batch[self.input_key], self._targets(batch[self.target_key])
        )
        rec = hierarchical_ce_loss(
            logits,
            self._targets(batch[self.target_key]),
            self.top_k_percentage,
            self.deterministic_top_k,
        )
        out = {
            f"kl_{i}": kl_divergence(qi, pi).flatten(1).sum(1).mean()
            for i, (qi, pi) in enumerate(zip(q, p, strict=True))
        }
        kl_sum = torch.stack(list(out.values())).sum()
        out.update(
            rec_loss_sum=rec["sum"],
            rec_loss_mean=rec["mean"],
            kl_sum=kl_sum,
            reconstruction=logits,
        )
        if self.loss_type == "geco":
            constraint, ma = self._geco_constraint(
                rec["sum"], rec["mask"].sum(1).mean()
            )
            out.update(
                loss=self.log_lagmul.exp() * constraint + kl_sum,
                constraint=constraint,
                ma=ma,
                lagmul=self.log_lagmul.exp(),
            )
        else:
            out["loss"] = rec["sum"] + self.beta * kl_sum
        return out

    def _step(self, batch: dict[str, Tensor], stage: str) -> dict[str, Tensor]:
        out = self.compute_loss(batch)
        self.log_dict(
            {
                f"{stage}_{k}": v
                for k, v in out.items()
                if k not in {"reconstruction", "ma"}
            },
            batch_size=batch[self.input_key].shape[0],
        )
        target = self._targets(batch[self.target_key])
        target = target.squeeze(1).long() if self.task == "binary" else target.argmax(1)
        pred = out["reconstruction"].detach()
        if self.task == "binary":
            pred = pred.squeeze(1).sigmoid()
        getattr(self, f"{stage}_metrics")(pred, target)
        return out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Optimize reconstruction and KL, updating GECO state once per batch."""
        out = self._step(batch, "train")
        if self.loss_type == "geco" and self.training:
            self._update_geco(out["constraint"], out["ma"])
        return out["loss"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Score posterior reconstructions without updating GECO state."""
        return self._step(batch, "val")["loss"]

    def on_train_epoch_end(self) -> None:
        """Log and reset training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def sample(
        self,
        img: Tensor,
        mean: bool | Sequence[bool] = False,
        z_q: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """Sample prior logits [B, C, H, W] without storing batch state."""
        return self.model.sample(img, mean, z_q)

    def reconstruct(
        self, img: Tensor, seg_one_hot: Tensor, mean: bool | Sequence[bool] = False
    ) -> tuple[Tensor, list[Independent], list[Independent]]:
        """Return posterior logits and q/p distributions for NCHW images/targets."""
        return self.model.reconstruct(img, seg_one_hot, mean)

    def forward(self, X: Tensor) -> Tensor:
        """Sample one prior segmentation as [B, C, H, W] logits."""
        return self.sample(X)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Return probabilities, entropy and logits [B, C, H, W, num_samples]."""
        samples = torch.stack([self.sample(X) for _ in range(self.num_samples)], dim=-1)
        return process_segmentation_prediction(samples, task=self.task)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Evaluate prior predictions and include targets and auxiliary data."""
        out = self.predict_step(batch[self.input_key])
        target = self._targets(batch[self.target_key])
        target = target.squeeze(1).long() if self.task == "binary" else target.argmax(1)
        pred = out["pred"].squeeze(1) if self.task == "binary" else out["pred"]
        self.test_metrics(pred, target)
        out[self.target_key] = batch[self.target_key]
        return self.add_aux_data_to_dict(out, batch)

    def on_test_start(self) -> None:
        """Create the prediction directory when saving is requested."""
        self.pred_dir = os.path.join(self.trainer.default_root_dir, self.pred_dir_name)
        if self.save_preds:
            os.makedirs(self.pred_dir, exist_ok=True)

    def on_test_epoch_end(self) -> None:
        """Log and reset prior prediction metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Save test predictions when enabled."""
        if self.save_preds:
            save_image_predictions(outputs, batch_idx, self.pred_dir)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the network optimizer, keeping the GECO multiplier separate."""
        optimizer = self.optimizer(p for p in self.parameters() if p.requires_grad)
        if self.lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler(optimizer),
                    "monitor": "val_loss",
                },
            }
        return {"optimizer": optimizer}
