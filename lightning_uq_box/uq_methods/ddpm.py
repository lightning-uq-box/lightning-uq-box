# Uses https://github.com/lucidrains/denoising-diffusion-pytorch/
# MIT License
# Copyright (c) 2020 Phil Wang

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Implement a Lightning Module Wrapper for Diffusion."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.guided_diffusion import (
    GaussianDiffusion as GuidedGaussianDiffusion,
)
from ema_pytorch import EMA
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import BaseModule


def classifier_cond_fn(
    x: Tensor, t: Tensor, classifier: nn.Module, y: Tensor, classifier_scale: float = 1
):
    """Compute gradient of classifier output wrt input.

    This is formally expressed as d_log(classifier(x, t)) / dx

    Args:
        x: input tensor
        t: timestep tensor
        classifier: classifier model
        y: target tensor
        classifier_scale: scale gradient by this factor anything greater than
            1 strengthens the classifier guidance. reportedly 3-8 is good empirically

    Returns:
        gradient of classifier output wrt input
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
        return grad


class DDPM(BaseModule):
    """Denoising Diffusion Probabilistic Model (DDPM).

    This trains a simple diffusion model.
    """

    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> Any:
        """Initialize a new instance of DDPM.

        Args:
            diffusion_model: diffusion model
            ema_decay: exponential moving average decay
            ema_update_every: update EMA every this many update calls
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        self.diffusion_model = diffusion_model
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = EMA(
            self.diffusion_model,
            beta=self.ema_decay,
            update_every=self.ema_update_every,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # lucidrains implementation does normalization, have to be careful

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch_size, device = (
            batch[self.input_key].shape[0],
            batch[self.input_key].device,
        )
        t = torch.randint(
            0, self.diffusion_model.num_timesteps, (batch_size,), device=device
        ).long()
        loss = self.diffusion_model.p_losses(batch[self.input_key], t)

        self.log("train_loss", loss)

        return loss

    def on_after_backward(self):
        """Update EMA after each backward pass."""
        self.ema.update()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        pass

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pass

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step that yields Diffusion Samples.

        Args:
            X: prediction batch of shape [batch_size x input_dims] for which
                to generate samples
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        batch_size = X.shape[0]
        sampled_images = self.diffusion_model.sample(
            batch_size, return_all_timesteps=True
        )

        return {"sample": sampled_images}

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.diffusion_model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class GuidedDDPM(DDPM):
    """Guided Diffusion Probabilistic Model (Guided-DDPM).

    Trains a classifier based on a pretrained Diffusion Model.
    """

    def __init__(
        self,
        diffusion_model: GuidedGaussianDiffusion,
        classifier: nn.Module,
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> Any:
        """Initialize a new instance of Guided DDPM."""
        super().__init__(
            diffusion_model, ema_decay, ema_update_every, optimizer, lr_scheduler
        )

        self.classifier = classifier

        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        # guided classifier training: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_train.py#L104 # noqa: E501
        batch_size, device = (
            batch[self.input_key].shape[0],
            batch[self.input_key].device,
        )
        t = torch.randint(
            0, self.diffusion_model.num_timesteps, (batch_size,), device=device
        ).long()

        q_sample = self.diffusion_model.q_sample(batch[self.input_key], t)

        logits = self.classifier(q_sample, t)

        loss = self.loss_fn(logits, batch[self.target_key])

        self.log("train_loss", loss)

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step with classifier guidance.

        Args:
            X: prediction batch of shape [batch_size x input_dims] for which
                to generate samples
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        batch_size = X.shape[0]
        sampled_images = self.diffusion_model.sample(
            batch_size,
            return_all_timesteps=True,
            cond_fn=classifier_cond_fn,
            guidance_kwargs={
                "classifier": self.classifier,
                "y": torch.fill(torch.zeros(batch_size), 1).long(),
                "classifier_scale": 1,
            },
        )

        return {"sample": sampled_images}

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
