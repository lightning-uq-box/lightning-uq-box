# Uses https://github.com/lucidrains/denoising-diffusion-pytorch/
# MIT License
# Copyright (c) 2020 Phil Wang

# adapted denoising-diffusion-pytorch to be a lightning module
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Denoising Diffusion Probabilistic Model (DDPM) and Latent DDPM for training on data beyond RGB."""

from typing import Any
import warnings

import os
from functools import partial
from random import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from torchvision.transforms import Normalize
from ema_pytorch import EMA
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from .base import BaseModule
from .utils import _get_num_outputs, default_classification_metrics


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def _to_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)


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
        model: nn.Module,
        betas: Tensor,
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        image_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 3,
        loss_fn: nn.Module = nn.MSELoss(),
        clip_denoised: bool = True,
        model_kwargs_keys: list[str] = [],
        log_samples_every_n_epochs: int = 10,
        latent_model: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize the DDPM model.

        Args:
            model: The denoising model, common choices are Unet architectures such as
                Generally expects a forward pass method that accepts the arguments
                (x, t, **model_kwargs) where x is the input image, t is the time step,
                and model_kwargs are additional arguments, in that order.
            betas: Noise scheduling betas for the diffusion process, shape (num_timesteps,).
                For different noise schedules, see `this file <../ddpm_utils.py>`_.
            ema_decay: The exponential moving average decay.
            ema_update_every: How often to update the EMA model, in terms of every n gradient steps.
            image_size: The size of the input images when coming from the dataloader, and also the size
                of the images that will be generated during sampling
            in_channels: The number of input channels of the images coming from the dataloader
            out_channels: The number of output channels of the images to be generated
            loss_fn: The loss function to use. By default, the loss function is called with the model prediction
                and target. If you need to pass additional arguments or need more control, you can override
                :method:`~DDPM.compute_loss` method.
            clip_denoised: Whether to clip the denoised image to be in the range [-1, 1] after each time step, makes
                sense for modalities with fixed ranges like RGB imagery, and use of Pretrained RGB Latent models.
            model_kwargs_keys: The names of additional keyword arguments that the forward pass of the model
                can accept. The keys will be used to extract the values from the batch dictionary, and
                passed to the model as keyword arguments.
            log_samples_every_n_epochs: The number of epochs between logging samples.
            latent_model: Optional latent model to encode the data,
                before running the diffusion process
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
        """
        super().__init__()
        self.model = model
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.image_size = _to_tuple(image_size)
        self.log_samples_every_n_epochs = log_samples_every_n_epochs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_kwargs_keys = model_kwargs_keys
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_denoised = clip_denoised

        self.latent_model = latent_model
        if self.latent_model:
            for param in self.latent_model.parameters():
                param.requires_grad = False
            self.latent_model.eval()

        # setup noise scheduling terms
        self.define_noise_scheduling_terms(betas)

        self.use_ema_model = False

        self.ema = EMA(model, beta=self.ema_decay, update_every=self.ema_update_every)

        self.loss_fn = loss_fn

        # TODO
        self.condition = False
        self.self_condition = False
        self.latent_scale_factor = 1.0

    def extract(self, a, t, x_shape) -> Tensor:
        """Extract the noise scheduling terms onto input tensor."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def define_noise_scheduling_terms(self, betas: Tensor) -> None:
        """Define the noise scheduling terms for the diffusion process.

        Args:the model_kwargs that are expected by the model.
            betas: Noise scheduling betas for the diffusion process, shape (num_timesteps,).
        """
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # TODO
        register_buffer("loss_weight", torch.ones_like(betas))

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        """Predict the noise from the starting image.

        Args:
            x_t (Tensor): The noisy image at time t.
            t (Tensor): The time step.
            x0 (Tensor): The predicted starting image.
        """
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the posterior distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start (Tensor): The starting image.
            x_t (Tensor): The noisy image at time t.
            t (Tensor): The time step.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The posterior mean, variance, and log variance.
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.inference_mode()
    def p_sample(
        self, x: Tensor, t: int, model_kwargs: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample from the model at a given time step.

        Args:
            x (Tensor): The noisy image at time t.
            t (int): The time step.
            x_self_cond (Tensor | None, optional): Self-conditioning tensor. Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: The predicted image and the starting image.
        """
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            model_kwargs=model_kwargs,
            clip_denoised=(self.latent_model is None) and self.clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        if t != 0:
            # i.e if train model with multiple outputs, then input during backward pass
            # should be the mean/median output
            pred_img = self.adapt_output_for_metrics(pred_img)
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: tuple[int, ...],
        return_all_timesteps: bool = False,
        model_kwargs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Run the sampling loop to generate images.

        Args:
            shape (tuple[int, ...]): The shape of the generated images.
            return_all_timesteps (bool, optional): Whether to return all timesteps. Defaults to False.
            cond_variables (Tensor | None, optional): model_kwargs for example conditioning variables. Defaults to None.

        Returns:
            Tensor: The generated images.
        """
        _, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # self_cond = cond_variables if self.cond_variables else None
            img, x_start = self.p_sample(img, t, model_kwargs)
            imgs.append(img)

        samples = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        if self.latent_model:
            samples = self.decode_latent(samples)
            # clip to [-1, 1] which is the input space of Latent encoder and normalize to [0, 1] for visualizaion
            samples = (samples.clamp(-1, 1) + 1) / 2

        # ret = self.unnormalize(ret)
        return samples

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int = 16,
        return_all_timesteps: bool = False,
        model_kwargs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Generate samples from the model.

        Args:
            batch_size (int, optional): The number of samples to generate. Defaults to 16.
            return_all_timesteps (bool, optional): Whether to return all timesteps. Defaults to False.
            cond_variables (Tensor | None, optional): Conditioning variables. Defaults to None.

        Returns:
            Tensor: The generated samples.
        """
        self.use_ema_model = True
        (h, w), channels = self.image_size, self.out_channels
        batch_size = (
            cond_variables.shape[0] if cond_variables is not None else batch_size
        )
        samples = self.p_sample_loop(
            (batch_size, channels, h, w),
            return_all_timesteps=return_all_timesteps,
            model_kwargs=model_kwargs,
        )
        self.use_ema_model = False
        return samples

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        """Sample from the forward process q(x_t | x_0).

        Args:
            x_start (Tensor): The starting image.
            t (Tensor): The time step.
            noise (Tensor | None, optional): The noise to add. Defaults to None.

        Returns:
            Tensor: The noisy image at time t.
        """
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(
        self, x, t, model_kwargs: dict[str, Tensor] | None = None, clip_x_start=False
    ):
        """Model predictions for the given input."""
        if self.use_ema_model:
            model_output = self.ema.ema_model(x, t, **model_kwargs)
        else:
            model_output = self.model(x, t, **model_kwargs)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0)
            if clip_x_start
            else torch.nn.Identity()
        )

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return {"pred_noise": pred_noise, "pred_x_start": x_start}

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation, this can be useful
        when training a model with multiple outputs, for example NLL optimization, but only
        want to compute metrics on the mean/median output.

        Args:
            out: output from the model

        Returns:
            mean/median output
        """
        return out

    def p_mean_variance(
        self, x, t, model_kwargs: dict[str, Tensor] | None = None, clip_denoised=True
    ):
        preds = self.model_predictions(x, t, **model_kwargs)
        x_start = preds["pred_x_start"]

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_losses(
        self,
        x_start,
        t,
        model_kwargs: dict[str, Tensor] | None = None,
        noise=None,
        offset_noise_strength=None,
    ):
        """Compute the losses for the model."""
        b, c, h, w = x_start.shape

        # In case there is a latent model, encode the image
        x_start = self.encode_img(self.latent_model, x_start, up_sample=False)

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        if self.self_condition and random() < 0.5:
            cond_variables = cond_variables
        else:
            cond_variables = None

        model_out = self.model(x, t, **model_kwargs)

        # Predict X_0
        return self.compute_loss(model_out, x_start, t, model_kwargs)

    def compute_loss(self, pred: Tensor, target: Tensor, t: Tensor, model_kwargs) -> Tensor:
        """Compute the loss for the model."""
        # TODO make this more varible, probably by also an argument
        # loss = F.mse_loss(pred, target, reduction="none")
        # loss = reduce(loss, "b ... -> b", "mean")
        # loss = loss * self.extract(self.loss_weight, t, loss.shape)
        return self.loss_fn(pred, target, **model_kwargs)

    def encode_img(
        self, latent_model: nn.Module, img: Tensor, up_sample: bool = False
    ) -> Tensor:
        """Upsample and encode to latent or just up sample low res image.

        Args:
            latent_model: The latent model to encode the image.
            img: The image to encode.
            up_sample: Whether to upsample the image.

        Returns:
            encoded image
        """
        # Raise warning and not error
        if img.min() < -1 or img.max() > 1:
            warnings.warn(
                "Input image should be normalized to be in range [-1, 1].", UserWarning
            )
        if up_sample:
            img = F.interpolate(img, scale_factor=self.upsample_factor, mode="bicubic")
        if latent_model:
            with torch.no_grad():
                img = latent_model.encode(img) * self.latent_scale_factor
        return img

    def decode_latent(self, z: Tensor) -> Tensor:
        """Decode latent diffusion model output to image."""
        with torch.no_grad():
            return torch.cat(
                [
                    self.latent_model.decode(z_.unsqueeze(0) / self.latent_scale_factor)
                    for z_ in z
                ],
                dim=0,
            )

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader with input and target key, conditioning tensors
                are expected to be found under the key 'conditions'.
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            training losslog_samples
        """
        X = batch[self.input_key]

        model_kwargs = {key: batch[key] for key in self.model_kwargs_keys}

        # target = batch[self.target_key]
        # fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(4, 16))
        # for i in range(8):
        #     axs[i, 0].imshow((X[i].permute(1, 2, 0).detach().cpu().numpy() + 1)/2)
        #     axs[i, 1].imshow((target[i].permute(1, 2, 0).detach().cpu().numpy()+1)/2)
        #     axs[i, 0].axis('off')
        #     axs[i, 1].axis('off')

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.trainer.default_root_dir, f"input_target_{self.current_epoch}.png"))
        # generate a t
        t = torch.randint(
            0, self.num_timesteps, (X.shape[0],), device=self.device
        ).long()
        loss = self.p_losses(X, t, **model_kwargs)

        self.log("train_loss", loss)
        return loss

    def on_after_backward(self):
        """Update EMA after each backward pass."""
        self.ema.update()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss."""
        # compute metrics on some valiation set

        # TODO maybe configure so that one batch of images is generated and saved
        # per log_samples_every_n_epochs
        if (
            self.current_epoch % self.log_samples_every_n_epochs == 0
            and batch_idx == 0
            and self.trainer.global_rank == 0
        ):
            samples = self.sample(batch_size=16)
            self.log_samples(samples, batch[self.input_key], batch.get("target", None))

    def log_samples(
        self, preds: Tensor, inputs: None | Tensor, targets: None | Tensor
    ) -> None:
        """Log samples to local directory.

        Args:
            preds: model predictions (samples)
            inputs: input data from data loader (if available)
            targets: target data from data loader (if available)
        """
        if hasattr(self.trainer.datamodule, "mean") and hasattr(
            self.trainer.datamodule, "std"
        ):
            mean = self.trainer.datamodule.mean
            std = self.trainer.datamodule.std
            if isinstance(mean, list):
                mean = torch.Tensor(mean).to(preds.device)
                std = torch.Tensor(std).to(preds.device)
            elif isinstance(mean, Tensor):
                mean = mean.to(preds.device)
                std = std.to(preds.device)
            elif isinstance(mean, float):
                mean = torch.Tensor([mean]).to(preds.device)
                std = torch.Tensor([std]).to(preds.device)
            preds = Normalize((-1 * mean / std), (1.0 / std))(preds).clamp(0, 1)
        elif preds.min() >= -1 and preds.max() <= 1:
            preds = (preds + 1) / 2

        rows, cols = preds.shape[0] // 4, 4
        _, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(preds[i].permute(1, 2, 0).cpu().numpy())
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.trainer.default_root_dir, f"sample_{self.current_epoch}.png"
            )
        )
        plt.close()

    def _scale_input(self, inputs: Tensor, t: Tensor) -> Tensor:
        """Scale the latent encodings."""
        if self.normalize_input:
            if self.latent_flag:
                # TODO normalize latent encodings
                inputs_norm = (inputs - 0.0) / 1.0
        else:
            inputs_norm = inputs
        return inputs_norm

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        # include Unet parameters
        optimizer = self.optimizer(self.model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "train_loss",
                    "interval": "step",
                },
            }
        else:
            return {"optimizer": optimizer}
