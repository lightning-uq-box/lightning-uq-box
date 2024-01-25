# This is a Lightning module wrapper for training a guided diffusion model
# with lightning instead of their custom trainer.
# It is based on the code from https://github.com/openai/guided-diffusion
# MIT License
# Copyright (c) 2021 OpenAI

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Guided Diffusion."""

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from lightning_uq_box.uq_methods import BaseModule
from lightning_uq_box.uq_methods.utils import default_classification_metrics


def my_create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    num_classes=None,
):
    """Create a UNetModel."""
    from guided_diffusion.unet import UNetModel

    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def my_create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    in_channels,
    out_channels,
):
    """Create a Classifier."""
    from guided_diffusion.unet import EncoderUNetModel

    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


# TODO the predict step probably foolows the image_sample.py script
class GuidedDiffusionModel(BaseModule):
    """Diffusion Model without Classifier.

    This implements the logic found in
    https://github.com/openai/guided-diffusion/blob/main/scripts/image_train.py
    in the lightning framework.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2105.05233
    """

    def __init__(
        self,
        image_size: int,
        model: Optional[nn.Module] = None,
        diffusion_process: Optional[nn.Module] = None,
        use_ddim: bool = False,
        diffusion_steps: int = 200,
        noise_schedule: str = "linear",
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new instance of Guided Diffusion model.

        Args:
            image_size: The size of the image.
            model: The diffusion model, for configuration see
            diffusion_process: The diffusion model, for configuration see
            use_ddim: whether or not to use Diffusion Denoising Implicit Models (DDIM)
                sampling strategy
            diffusion_steps: number of diffusion steps to take
            noise_schedule: what noise schedule to use, one of [`linear`, `cosine`]
            optimizer: The optimizer to use for training
            lr_scheduler: The learning rate scheduler to use for training
        """
        super().__init__()
        self.image_size = image_size
        self.model = model
        self.diffusion_process = diffusion_process

        self.loss_fn = nn.CrossEntropyLoss()

        self.use_ddim = use_ddim
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.build_model()

    def build_model(self) -> None:
        """Build the model."""
        try:
            import guided_diffusion.gaussian_diffusion as gd
            from guided_diffusion.resample import create_named_schedule_sampler
            from guided_diffusion.script_util import create_gaussian_diffusion

            def my_get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
                """Named beta schedules for the diffusion process."""
                if schedule_name == "linear":
                    beta_start = 1e-5
                    beta_end = 1e-2
                    return np.linspace(
                        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                    )
                elif schedule_name == "cosine":
                    return gd.betas_for_alpha_bar(
                        num_diffusion_timesteps,
                        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                    )
                else:
                    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

            gd.get_named_beta_schedule = my_get_named_beta_schedule
        except ImportError:
            raise ImportError(
                "guided_diffusion is not installed and is required to use this model. "
                "It can be installed with "
                "`pip install git+https://github.com/openai/guided-diffusion`."
            )
        if self.model is None:
            self.model = my_create_model(
                image_size=self.image_size,
                num_channels=128,
                num_res_blocks=2,
                channel_mult="",
                learn_sigma=False,
                use_checkpoint=False,
                attention_resolutions="16,8",
                num_heads=4,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                dropout=0.0,
                resblock_updown=False,
                use_fp16=False,
                use_new_attention_order=False,
                num_classes=None,
            )

        if self.diffusion_process is None:
            self.diffusion_process = create_gaussian_diffusion(
                learn_sigma=False,
                steps=self.diffusion_steps,
                noise_schedule=self.noise_schedule,
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
            )

        self.schedule_sampler = create_named_schedule_sampler(
            name="uniform", diffusion=self.diffusion_process
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
        # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/train_util.py#L172 # noqa E501
        t, weights = self.schedule_sampler.sample(
            batch[self.input_key].shape[0], batch[self.input_key].device
        )

        # need to understand what to condition on
        losses = self.diffusion_process.training_losses(
            model=self.model,
            x_start=batch[self.input_key],
            t=t,
            # model_kwargs={"y": batch[self.target_key]},
        )

        loss = (losses["loss"] * weights).mean()

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Validation step not necessary."""
        pass

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Test step not necessary, samples are yielded in predict_step."""
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
        # https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py # noqa E501
        sample_fn = (
            self.diffusion_process.p_sample_loop
            if not self.use_ddim
            else self.diffusion_process.ddim_sample_loop
        )

        sample = sample_fn(
            self.model,
            (X.shape[0], X.shape[1], self.image_size, self.image_size),
            clip_denoised=True,
            model_kwargs={},
        )

        return {"sample": sample}
        # also compute metrics?
        # https://github.com/openai/guided-diffusion/blob/main/scripts/image_nll.py # noqa E50155

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the lightning documentation
        """
        optimizer = self.optimizer(self.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


# TODO the predict step probably foolows the classifier_sample.py script
class GuidedDiffusionClassifier(BaseModule):
    """Guided Diffusion with Classifier Model.

    This is a Lightning module for training a guided diffusion model with a classifier.

    This implements the logic found in
    https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_train.py
    in the lightning framework.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2105.05233
    """

    valid_tasks = ["binary", "multiclass", "multilabel"]

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        classifier: Optional[nn.Module] = None,
        diffusion_process: Optional[nn.Module] = None,
        model: Optional[nn.Module] = None,
        use_ddim: bool = False,
        diffusion_steps: int = 200,
        noise_schedule: str = "linear",
        task: str = "multiclass",
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new instance of Guided Diffusion model.

        Args:
            image_size: The size of the image.
            num_classes: The number of classes the classifier model is trained on
            classifier: The diffusion model that can be unguided or guided, for configuration see
                https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L228 # noqa E501
                if not provided, will use default config
            diffusion_process: The diffusion model, for configuration see
                https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L386 # noqa E501
                if not provided, will use default config
            model: pretrained unconditional diffusion model, which a trained classifier will guide. Necessary for prediction step.
            use_ddim: whether or not to use Diffusion Denoising Implicit Models (DDIM)
                sampling strategy
            diffusion_steps: number of diffusion steps to take
            noise_schedule: what noise schedule to use, one of [`linear`, `cosine`]
            task: The task, one of [`multiclass`, `multilabel`, `binary`]
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler.

        """
        super().__init__()

        self.diffusion_process = diffusion_process
        self.classifier = classifier

        self.image_size = image_size

        self.use_ddim = use_ddim
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_fn = nn.CrossEntropyLoss()

        self.model = model

        self.task = task

        self.num_classes = num_classes

        self.build_model()

        self.setup_task()

    def build_model(self) -> None:
        """Build the model."""
        try:
            import guided_diffusion.gaussian_diffusion as gd
            from guided_diffusion.resample import create_named_schedule_sampler
            from guided_diffusion.script_util import create_gaussian_diffusion

            def my_get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
                """Named beta schedules for the diffusion process."""
                if schedule_name == "linear":
                    beta_start = 1e-5
                    beta_end = 1e-2
                    return np.linspace(
                        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                    )
                elif schedule_name == "cosine":
                    return gd.betas_for_alpha_bar(
                        num_diffusion_timesteps,
                        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                    )
                else:
                    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

            gd.get_named_beta_schedule = my_get_named_beta_schedule
        except ImportError:
            raise ImportError(
                "guided_diffusion is not installed and is required to use this model. "
                "It can be installed with "
                "."
            )

        if self.classifier is None:
            self.classifier = my_create_classifier(
                self.image_size,
                classifier_use_fp16=False,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions="32,16,8",  # 16
                classifier_use_scale_shift_norm=True,  # False
                classifier_resblock_updown=True,  # False
                classifier_pool="attention",
                out_channels=self.num_classes,
                in_channels=3,
            )

        if self.diffusion_process is None:
            self.diffusion_process = create_gaussian_diffusion(
                learn_sigma=False,
                steps=self.diffusion_steps,
                noise_schedule=self.noise_schedule,
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
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

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_train.py#L119 # noqa E501
        t, _ = self.schedule_sampler.sample(
            batch[self.input_key].shape[0], batch[self.input_key].device
        )
        X = self.diffusion_process.q_sample(batch[self.input_key], t=t)

        logits = self.classifier(X, timesteps=t)

        loss = self.loss_fn(logits, batch[self.target_key])

        self.train_metrics(logits, batch[self.target_key])

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
            batch[self.input_key].shape[0], batch[self.input_key].device
        )
        X = self.diffusion_process.q_sample(batch[self.input_key], t=t)

        logits = self.classifier(X, timesteps=t)

        loss = self.loss_fn(logits, batch[self.target_key])

        self.val_metrics(logits, batch[self.target_key])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Test step not necessary, samples are yielded in predict_step."""
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
        if self.model is None:
            raise ValueError(
                "model is None, cannot sample from diffusion process, please provide an unconditioned diffusion model"  # noqa E501
            )

        sample_fn = (
            self.diffusion_process.p_sample_loop
            if not self.use_ddim
            else self.diffusion_process.ddim_sample_loop
        )

        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in)[
                    0
                ]  # * args.classifier_scale

        def model_fn(x, t, y=None):
            assert y is not None
            return self.model(x, t, y)

        classes = torch.randint(
            low=0, high=self.num_classes, size=(X.shape[0],), device=X.device
        )
        model_kwargs = {"y": classes}

        sample = sample_fn(
            model_fn,
            (X.shape[0], X.shape[1], self.image_size, self.image_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=X.device,
        )

        return {"sample": sample}

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
