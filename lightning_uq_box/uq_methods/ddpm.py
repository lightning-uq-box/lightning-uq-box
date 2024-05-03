# Uses https://github.com/lucidrains/denoising-diffusion-pytorch/
# MIT License
# Copyright (c) 2020 Phil Wang

# adapted denoising-diffusion-pytorch to be a lightning module
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Implement a Lightning Module Wrapper for Diffusion."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
import matplotlib.pyplot as plt

from .base import BaseModule
from torchvision.utils import make_grid
from .utils import _get_num_outputs, default_classification_metrics

if TYPE_CHECKING:
    try:
        from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
            GaussianDiffusion,
        )
        from denoising_diffusion_pytorch.guided_diffusion import (
            GaussianDiffusion as GuidedGaussianDiffusion,
        )
    except ImportError:
        pass


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

    This trains a simple diffusion model based on the implementation
    of `denoising-diffusion-pytorch repo
    <https://github.com/lucidrains/denoising-diffusion-pytorch/>`_.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        diffusion_model: "GaussianDiffusion",
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        log_samples_every_n_steps: int = 1000,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> Any:
        """Initialize a new instance of DDPM.

        Args:
            diffusion_model: diffusion model
            ema_decay: exponential moving average decay
            ema_update_every: update EMA every this many update calls
            log_samples_every_n_steps: log samples every n steps
            optimizer: optimizer to use
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        try:
            import denoising_diffusion_pytorch  # noqa: F401
            from ema_pytorch import EMA
        except ImportError:
            raise ImportError(
                "You need to install the denoising-diffusion-pytorch package."
            )

        self.diffusion_model = diffusion_model
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = EMA(
            self.diffusion_model,
            beta=self.ema_decay,
            update_every=self.ema_update_every,
        )
        self.log_samples_every_n_steps = log_samples_every_n_steps
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(
        self, batch_size: int, return_all_timesteps: bool = True, **kwargs: Any
    ) -> Any:
        """Forward pass of DDPM model for inference is sampling."""
        return self.diffusion_model.sample(
            batch_size, return_all_timesteps=return_all_timesteps, 
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            training loss
        """
        loss = self.diffusion_model.forward(batch[self.input_key])

        self.log("train_loss", loss)

        if self.trainer.global_step % self.log_samples_every_n_steps == 0 and self.trainer.global_rank == 0:
            # log samples
            sampled_imgs = self.forward(16, return_all_timesteps=False).detach()
            fig, ax = plt.subplots(4, 4, figsize=(32, 32))
            for i in range(16):
                ax[i // 4, i % 4].imshow(sampled_imgs[i].cpu().numpy().transpose(1, 2, 0))
                ax[i // 4, i % 4].axis("off")
            plt.tight_layout()
            fig.savefig(self.trainer.default_root_dir + f"/sample_{self.trainer.global_step}.png")

        return loss

    def on_after_backward(self):
        """Update EMA after each backward pass."""
        self.ema.update()

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """No test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pass

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
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
            }
        else:
            return {"optimizer": optimizer}


class GuidedDDPM(DDPM):
    """Guided Diffusion Probabilistic Model (Guided-DDPM).

    Trains a classifier based on a pretrained Diffusion Model
    for the implementation
    of `denoising-diffusion-pytorch repo
    <https://github.com/lucidrains/denoising-diffusion-pytorch/>`_.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2105.05233
    """

    def __init__(
        self,
        diffusion_model: "GuidedGaussianDiffusion",
        classifier: nn.Module,
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> Any:
        """Initialize a new instance of Guided DDPM."""
        super().__init__(
            diffusion_model, ema_decay, ema_update_every, optimizer, lr_scheduler
        )

        self.classifier = classifier

        self.num_classes = _get_num_outputs(self.classifier)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_metrics = default_classification_metrics(
            "train", "multiclass", self.num_classes
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: index of batch
            dataloader_idx: index of dataloader

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
        self.train_metrics(logits, batch[self.target_key])

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def forward(
        self,
        batch_size: int,
        classifier_scale: float = 1.0,
        cond_fn=classifier_cond_fn,
        y: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass of Guided DDPM model for inference is sampling.

        Args:
            batch_size: batch size
            classifier_scale: scale the classifier guidance gradient
            cond_fn: conditional function
            y: conditional classes tensor
            kwargs: arguments for guidance

        Returns:
            sampled images
        """
        if y is None:
            y = torch.randint(
                0, self.num_classes, (batch_size,), device=self.device
            ).long()
        return self.diffusion_model.sample(
            batch_size,
            return_all_timesteps=True,
            cond_fn=cond_fn,
            guidance_kwargs={
                "classifier": self.classifier,
                "y": y,
                "classifier_scale": classifier_scale,
            },
        )

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
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
            }
        else:
            return {"optimizer": optimizer}


class ClassFreeGuidanceDDPM(DDPM):
    """Classifier free Guidance DDPM.

    This trains a classifier free guidance diffusion model based on the implementation
    of `denoising-diffusion-pytorch repo
    <https://github.com/lucidrains/denoising-diffusion-pytorch/>`_.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2207.12598
    """

    def __init__(
        self,
        diffusion_model: "GaussianDiffusion",
        num_classes: int,
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> Any:
        """Initialize a new instance of Guidance Free DDPM.

        Args:
            diffusion_model: diffusion model
            num_classes: number of classes
            ema_decay: exponential moving average decay
            ema_update_every: update EMA every this many update calls
            optimizer: optimizer
            lr_scheduler: learning rate scheduler
        """
        super().__init__(
            diffusion_model, ema_decay, ema_update_every, optimizer, lr_scheduler
        )
        self.num_classes = num_classes
        assert (
            self.diffusion_model.model.cond_drop_prob > 0
        ), "cond_prob_drop is 0, but for guidance free training it should be > 0"

    def forward(
        self,
        classes: Tensor,
        cond_scale: float = 6.0,
        rescaled_phi: float = 0.7,
        **kwargs: Any,
    ) -> Any:
        """Forward pass of DDPM model for inference is sampling.

        Args:
            classes: target classes
            cond_scale: scale the conditional where values > 1 strengthen
                the classifier free guidance
            rescaled_phi: rescale phi
            kwargs: additional arguments for diffusion sampling

        """
        return self.diffusion_model.sample(
            classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi
        )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            training loss
        """
        loss = self.diffusion_model.forward(
            batch[self.input_key], classes=batch[self.target_key]
        )

        self.log("train_loss", loss)

        return loss
