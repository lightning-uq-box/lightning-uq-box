# This is a Lightning module wrapper for training a guided diffusion model
# with lightning instead of their custom trainer.
# It is based on the code from https://github.com/openai/guided-diffusion
# MIT License
# Copyright (c) 2021 OpenAI

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Guided Diffusion."""

from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.uq_methods import BaseModule
from lightning_uq_box.uq_methods.utils import default_classification_metrics


class GuidedDiffusionClassifier(BaseModule):
    """Guided Diffusion.

    This is a Lightning module for training a guided diffusion model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2105.05233
    """

    def __init__(
        self,
        image_size: int,
        classifier: Optional[nn.Module] = None,
        diffusion_process: Optional[nn.Module] = None,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new instance of Guided Diffusion model.

        Args:
            image_size: The size of the image.
            classifier: The classifier model, for configuration see
                https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L228
                if not provided, will use default config
            diffusion_process: The diffusion model, for configuration see
                https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L386
                if not provided, will use default config
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler.

        """
        super().__init__()

        self.diffusion_process = diffusion_process
        self.classifier = classifier

        self.image_size = image_size

        self.build_model()

        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self) -> None:
        """Build the model."""
        try:
            from guided_diffusion.resample import create_named_schedule_sampler
            from guided_diffusion.script_util import (
                create_classifier,
                create_gaussian_diffusion,
            )
        except ImportError:
            raise ImportError(
                "guided_diffusion is not installed and is required to use this model. "
                "It can be installed with "
                "`pip install git+https://github.com/openai/guided-diffusion`."
            )
        if self.diffusion_process is None:
            self.diffusion_process = create_gaussian_diffusion(
                learn_sigma=False,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
            )

        if self.classifier is None:
            self.classifier = create_classifier(
                self.image_size,
                classifier_use_fp16=False,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions="32,16,8",  # 16
                classifier_use_scale_shift_norm=True,  # False
                classifier_resblock_updown=True,  # False
                classifier_pool="attention",
            )

        self.schedule_sampler = create_named_schedule_sampler(
            name="uniform", diffusion=self.diffusion_process
        )

    def setup_task(self) -> None:
        """Setup the task."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        t, _ = self.schedule_sampler.sample(
            batch["image"].shape[0], batch["image"].device
        )
        batch = self.diffusion_process.q_sample(batch, timesteps=t)

        logits = self.classifier(batch["image"], timesteps=t)

        loss = self.loss_fn(logits, batch["label"])

        self.train_metrics(logits, batch["label"])

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        t, _ = self.schedule_sampler.sample(
            batch["image"].shape[0], batch["image"].device
        )
        batch = self.diffusion_process.q_sample(batch, timesteps=t)

        logits = self.classifier(batch["image"], timesteps=t)

        loss = self.loss_fn(logits, batch["label"])

        self.val_metrics(logits, batch["label"])

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the lightning documentation
        """
        optimizer = self.optimizer(self.classifier.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
