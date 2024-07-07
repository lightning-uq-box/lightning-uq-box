# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Variational Auto Encoder (VAE)."""

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam as Adam

from .base import DeterministicPixelRegression
from .loss_functions import VAELoss
from lightning.pytorch.utilities import rank_zero_only
import matplotlib.pyplot as plt
from lightning_uq_box.models import VAEDecoder
from .utils import _get_num_inputs, freeze_segmentation_model


class VAE(DeterministicPixelRegression):
    """Variational Auto Encoder (VAE) for Torchseg Encoders.

    This VAE is intended to be used with
    `Torchseg Encoders <https://github.com/isaaccorley/torchseg/blob/main/torchseg/encoders/timm.py>`__
    that support a wide range og `Timm Models <https://rwightman.github.io/pytorch-image-models/>`__.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_size: int,
        num_samples: int,
        out_channels: int,
        img_size: int,
        loss_fn: nn.Module | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the VAE.

        Args:
            encoder: Encoder Timm Model
            latent_size: The size of the latent space.
            num_samples: The number of samples to draw from the latent space for prediction.
            out_channels: The number of output channels.
            img_size: The size of the input image, needed to configure the decoder and
                sampling procedure
            loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        if loss_fn is None:
            loss_fn = VAELoss()

        self.out_channels = out_channels
        self.latent_size = latent_size
        self.num_samples = num_samples
        self.img_size = img_size

        super().__init__(
            None,
            loss_fn,
            freeze_backbone,
            freeze_decoder,
            optimizer,
            lr_scheduler,
            save_preds,
        )

        self.encoder = encoder
        self.configure_latent_net()

        self.freeze_model()

    @property
    def num_input_features(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        return _get_num_inputs(self.encoder)

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of output dimension to model
        """
        return self.out_channels

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.conv_mu.parameters():
                param.requires_grad = True
            for param in self.conv_log_var.parameters():
                param.requires_grad = True

        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.conv_init_decoder.parameters():
                param.requires_grad = False

    def configure_latent_net(self) -> None:
        """Configure the latent net and decoder."""
        # decoder_channels = [info["num_chs"] for info in self.encoder.feature_info][::-1]
        decoder_channels = self.encoder.out_channels[::-1]
        # replace last decoder channel with final output channel
        decoder_channels[-1] = self.out_channels

        latent_channels = decoder_channels[0]

        self.latent_image_dim = self.img_size // self.encoder.reductions[-1]

        self.conv_mu = nn.Conv2d(latent_channels, self.latent_size, 1, 1)
        self.conv_log_var = nn.Conv2d(latent_channels, self.latent_size, 1, 1)

        self.conv_init_decoder = nn.Conv2d(self.latent_size, latent_channels, 1, 1)

        self.decoder = VAEDecoder(decoder_channels=decoder_channels)

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for metrics."""
        return out

    def reparameterization_trick(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for the VAE.

        Args:
            mu: The mean tensor.
            log_var: The log variance tensor.

        Returns:
            The reparameterized tensor.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the VAE.

        Args:
            x: The input tensor
            conditions: The conditions tensor

        Returns:
            The output tensor.
        """

        x_encoded = self.encoder.forward(x)[-1]

        # encode the latent space
        mu = self.conv_mu(x_encoded)
        log_var = self.conv_log_var(x_encoded)
        z = self.reparameterization_trick(mu, log_var)

        x_decoder_init = self.conv_init_decoder(z)

        # decode
        x_recon = self.decoder(x_decoder_init)

        return x_recon, mu, log_var

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        """Training step for the VAE.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            The loss and the log dictionary.
        """
        X, y = batch[self.input_key], batch[self.target_key]
        batch_size = X.shape[0]

        x_recon, mu, log_var = self.forward(X)

        scaled_kl_loss, rec_loss = self.loss_fn(x_recon, y, mu, log_var)

        self.log("train_kl_loss", scaled_kl_loss, batch_size=batch_size)
        self.log("train_rec_loss", rec_loss, batch_size=batch_size)
        self.log("train_loss", scaled_kl_loss + rec_loss, batch_size=batch_size)

        self.train_metrics(x_recon, y)

        return scaled_kl_loss + rec_loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Tensor]:
        """Validation step for the VAE.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        """
        X, y = batch[self.input_key], batch[self.target_key]
        batch_size = X.shape[0]

        x_recon, mu, log_var = self.forward(X)

        scaled_kl_loss, rec_loss = self.loss_fn(x_recon, y, mu, log_var)

        self.log("val_kl_loss", scaled_kl_loss, batch_size=batch_size)
        self.log("val_rec_loss", rec_loss, batch_size=batch_size)
        self.log("val_loss", scaled_kl_loss + rec_loss, batch_size=batch_size)

        self.val_metrics(x_recon, y)

        if self.trainer.global_step % 1 == 0 and self.trainer.global_rank == 0:
            # log samples
            self.plot_and_save_samples(batch)

        return scaled_kl_loss + rec_loss

    @rank_zero_only
    def plot_and_save_samples(self, batch: dict[str, Tensor]) -> None:
        """Plot samples from VAE model."""
        with torch.no_grad():
            sampled_imgs = self.sample(num_samples=16).detach()

        fig, ax = plt.subplots(4, 4, figsize=(32, 32))
        for i in range(16):
            ax[i // 4, i % 4].imshow(sampled_imgs[i].cpu().numpy().transpose(1, 2, 0))
            ax[i // 4, i % 4].axis("off")
        plt.tight_layout()
        fig.savefig(
            self.trainer.default_root_dir + f"/sample_{self.trainer.global_step}.png"
        )

    def predict_step(
        self, X: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction Step with VAE.

        Args:
            X: The input tensor.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            Prediction dictionary
        """
        x_encoded = self.encoder(X)[-1]

        # encode the latent space once
        mu = self.conv_mu(x_encoded)
        log_var = self.conv_log_var(x_encoded)

        # sample latent space and decode
        self.predictions: list[Tensor] = []
        for _ in range(self.num_samples):
            z = self.reparameterization_trick(mu, log_var)
            x = self.conv_init_decoder(z)
            x = self.decoder(x)
            self.predictions.append(x)

        preds = torch.stack(self.predictions, dim=-1)

        return {"pred": preds.mean(dim=-1), "pred_uct": preds.std(dim=-1)}

    def sample(self, num_samples: int = 16) -> Tensor:
        """Sample from the VAE.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            The samples.
        """
        z = torch.randn(
            num_samples, self.latent_size, self.latent_image_dim, self.latent_image_dim
        ).to(self.device)
        x_decoded = self.decoder(self.conv_init_decoder(z))
        return x_decoded


# class ConditionalVAE(VAE):
#     """Conditional Variational Auto Encoder for Torchseg."""

#     def __init__(
#         self,
#         model: nn.Module,
#         latent_size: int,
#         loss_fn: nn.Module = VAELoss,
#         num_samples: int = 5,
#         freeze_backbone: bool = False,
#         freeze_decoder: bool = False,
#         optimizer: OptimizerCallable = torch.optim.Adam,
#         lr_scheduler: LRSchedulerCallable = None,
#         save_preds: bool = False,
#     ) -> None:
#         """Initialize the Conditional VAE.

#         Args:
#             model: Torchseg model.
#             latent_size: The size of the latent space.
#             loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`.
#             num_samples: The number of samples to draw from the latent space for prediction.
#             freeze_backbone: Whether to freeze the backbone.
#             freeze_decoder: Whether to freeze the decoder.
#             optimizer: The optimizer to use.
#             lr_scheduler: The learning rate scheduler.
#             save_preds: Whether to save predictions.
#         """
#         super().__init__(
#             model,
#             loss_fn,
#             freeze_backbone,
#             freeze_decoder,
#             optimizer,
#             lr_scheduler,
#             save_preds,
#         )
