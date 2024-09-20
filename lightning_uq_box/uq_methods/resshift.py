# Based on https://github.com/zsyOAOA/ResShift
# S-Lab License 1.0
# Copyright 2022 S-Lab

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.


from .ddpm_scratch import DDPM
from torch import Tensor
import torch
import matplotlib.pyplot as plt

import os
import torch.nn as nn

from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from tqdm import tqdm
import numpy as np


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class ResShiftSR(DDPM):
    """Super Resolution with Residual Shift."""

    def __init__(
        self,
        model: nn.Module,
        betas: Tensor, 
        ema_decay: float = 0.995,
        ema_update_every: float = 10,
        image_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 3,
        kappa: float = 1.0,
        super_res_factor: int = 4,
        log_samples_every_n_epochs: int = 10,
        latent_model: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize the DDPM model.

        Args:
            model: The denoising model, common choices are Unet architectures such as

            betas: Noise scheduling betas for the diffusion process, shape (num_timesteps,).
                For different noise schedules, see 
            ema_decay: The exponential moving average decay.
            ema_update_every: The number of steps to update the EMA.
            image_size: The size of the input images when coming from the dataloader, and also the size
                of the images that will be generated during sampling
            input_channels: The number of input channels of the images coming from the dataloader
            output_channels: The number of output channels of the images to be generated
            kappa: The kappa value for the noise schedule.
            super_res_factor: The super resolution factor.
            log_samples_every_n_epochs: The number of epochs between logging samples.
            latent_model: Optional latent model to encode the data,
                before running the diffusion process
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler to use.
        """
        self.kappa = kappa
        self.super_res_factor = super_res_factor
        self.upsample_factor = super_res_factor
        super().__init__(
            model = model,
            betas = betas,
            ema_decay = ema_decay,
            ema_update_every = ema_update_every,
            image_size = image_size,
            in_channels = in_channels,
            out_channels = out_channels,
            log_samples_every_n_epochs = log_samples_every_n_epochs,
            latent_model = latent_model,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
        )

        self.normalize_input = True
        self.latent_flag = (latent_model is not None)


    def define_noise_scheduling_terms(self, betas: Tensor) -> None:
        """Define noise scheduling terms."""
        super().define_noise_scheduling_terms(betas)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('sqrt_betas', torch.sqrt(betas))
        self.betas_prev = np.append(0.0, self.betas[:-1])
        posterior_variance = self.kappa**2 * self.betas_prev / self.betas * (self.betas - self.betas_prev)
        posterior_variance_clipped = torch.cat(
            (posterior_variance[1].unsqueeze(0), posterior_variance[1:])
        )
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance_clipped.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', self.betas_prev / self.betas)
        register_buffer('posterior_mean_coef2', (self.betas - self.betas_prev) / self.betas)

    def q_sample(self, x_start, lr_img, t, noise = None):
        """Q sample q(x_t | x_0}) with shift based on lr_img"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * (lr_img - x_start) + x_start +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, x, t, lr_img: Tensor, clip_x_start = False):
        """Model predictions for the given input."""
        if self.use_ema_model:
            model_output = self.ema.ema_model(self._scale_input(x, t), t, lr_img)
        else:
            model_output = self.model(self._scale_input(x, t), t, lr_img)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else torch.nn.Identity()

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return {
            "pred_noise": pred_noise,
            "pred_x_start": x_start,
        }

    def p_mean_variance(self, x, lr_img: Tensor, t, clip_denoised = True):
        preds = self.model_predictions(x, t, lr_img)
        x_start = preds["pred_x_start"]

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, lr_img: Tensor, t: int):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, lr_img=lr_img, t = batched_times, clip_denoised = (self.latent_model is None))
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_losses(self, x_start, lr_img, t, noise = None, offset_noise_strength = None):
        """Compute the prediction loss.
        
        Args:
            x_start: The image to predict, in the case of super resolution, this is the HR image.
            lr_img: The low resolution image.
            t: The time step.
            noise: The noise to use for the sampling.
            offset_noise_strength: The noise strength to use for the offset.
        """
        b, c, h, w = x_start.shape

        # HR Image gets encoded to be the prediction target
        x_start = self.encode_img(self.latent_model, x_start, up_sample = False)
            
        noise = default(noise, lambda: torch.randn_like(x_start))

        # for q_sampling the lr_img is also encoded
        # https://github.com/zsyOAOA/ResShift/blob/dfc2ff705a962de1601a491511b43a93b97d9622/models/gaussian_diffusion.py#L555
        x = self.q_sample(x_start = x_start, lr_img =self.encode_img(self.latent_model, lr_img, up_sample = True), t = t, noise = noise)

        # predict and take gradient step
        # for the model input, the lr_img is not encoded
        # see https://github.com/zsyOAOA/ResShift/blob/dfc2ff705a962de1601a491511b43a93b97d9622/models/gaussian_diffusion.py#L566 model_kwargs
        model_out = self.model(self._scale_input(x, t), t, lr_img)

        target = x_start
        return self.compute_loss(model_out, target, t)

    def prior_sample(self, y, noise=None):
        """
        :param y: the [N x C x ...] tensor of degraded inputs, encoded LR image
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        return y + self.extract(self.kappa * self.sqrt_betas, t, y.shape) * noise


    def p_sample_loop(self, lr_img: Tensor, return_all_timesteps = False):
        batch, device = lr_img.shape[0], self.device
        
        z_y = self.encode_img(self.latent_model, lr_img, up_sample = True)

        # img to start sampling process with
        img = self.prior_sample(z_y, torch.randn_like(z_y))
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, lr_img, t)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        if self.latent_model:
            ret = self.decode_latent(ret)

        # clamp and normalize to 0, 1
        ret = (ret.clamp(-1, 1) + 1) / 2

        return ret


    def _scale_input(self, inputs: Tensor, t: Tensor) -> Tensor:
        """Scale the latent encodings."""
        if self.normalize_input:
            if self.latent_flag:
                # the variance of latent code is around 1.0
                std = torch.sqrt(self.extract(self.betas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = self.extract(self.sqrt_betas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def q_posterior(self, x_start, x_t, t):
        # print("Q Posterior Mean Variance")
        # print(x_t.min(), x_t.max(), x_t.mean(), x_t.std())
        
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        # print("After Q Posterior Mean Variance")
        # print(posterior_mean.min(), posterior_mean.max(), posterior_mean.mean(), posterior_mean.std())
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def training_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute and return the training loss."""
        # compute the loss
        lr_img, hr_img = batch[self.input_key], batch[self.target_key]

        t = torch.randint(0, self.num_timesteps, (lr_img.shape[0],), device = self.device).long()
        loss = self.p_losses(hr_img, lr_img, t)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute and return the validation loss."""
        # compute metrics on some valiation set
        
        # TODO maybe configure so that one batch of images is generated and saved
        # per log_samples_every_n_epochs
        if self.current_epoch % self.log_samples_every_n_epochs == 0 and batch_idx == 0 and self.trainer.global_rank == 0:
            preds = self.predict_step(batch[self.input_key], return_all_timesteps = False)
        #     samples = self.sample(batch_size = 16)
            self.log_samples(preds["pred"], preds["lr"], batch[self.target_key])

    def log_samples(self, preds: Tensor, inputs: Tensor, targets: Tensor) -> None:
        """Log samples to local directory.
        
        Args:
            preds: model predictions (samples)
            inputs: input data from data loader (if available)
            targets: target data from data loader (if available)
        """
       # normalize inputs to 0, 1 for plotting
        inputs = (inputs + 1) / 2
        targets = (targets + 1) / 2

        rows, cols = preds.shape[0] // 4, 4
        batch_size = preds.shape[0]
        # sample random img indices
        random_idxs = np.random.choice(range(batch_size), 8, replace=False)
        _, axs = plt.subplots(len(random_idxs), 3, figsize=(cols * 2, rows * 2))
        for axs_idx, i in enumerate(random_idxs):
            axs[axs_idx, 0].imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))
            axs[axs_idx, 0].axis("off")
            axs[axs_idx, 1].imshow(preds[i].cpu().numpy().transpose(1, 2, 0))
            axs[axs_idx, 1].axis("off")
            axs[axs_idx, 2].imshow(targets[i].cpu().numpy().transpose(1, 2, 0))
            axs[axs_idx, 2].axis("off")
        
        axs[0, 0].set_title("LR Image")
        axs[0, 1].set_title("Prediction")
        axs[0, 2].set_title("HR Image")

        plt.tight_layout()
        plt.savefig(os.path.join(self.trainer.default_root_dir, f"sample_{self.current_epoch}.png"))
        plt.close()


    def predict_step(self, lr_img: Tensor, return_all_timesteps: bool = False) -> dict[str, Tensor]:
        """Predict the super resolved image."""
        hr_imgs = self.p_sample_loop(lr_img, return_all_timesteps = return_all_timesteps)

        return {"pred": hr_imgs, "lr": lr_img}
        

# class InpaintingResShift(DDPM)