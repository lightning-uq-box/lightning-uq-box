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


class VAE(DeterministicPixelRegression):
    """Variational Auto Encoder (VAE) for Torchseg model."""

    def __init__(
        self,
        model: nn.Module,
        latent_size: int,
        loss_fn: nn.Module | None = None,
        num_samples: int = 5,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the VAE.

        Args:
            model: Torchseg model.
            latent_size: The size of the latent space.
            loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`.
            num_samples: The number of samples to draw from the latent space for prediction.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        if loss_fn is None:
            loss_fn = VAELoss()
        super().__init__(
            model,
            loss_fn,
            freeze_backbone,
            freeze_decoder,
            optimizer,
            lr_scheduler,
            save_preds,
        )
        self.latent_size = latent_size
        self.num_samples = num_samples

        self.configure_latent_net()

    def configure_latent_net(self) -> None:
        """Configure the latent net that encodes the latent space from encoder output."""
        out_channels = self.model.encoder.out_channels[-1]

        self.conv_mu = nn.Conv2d(out_channels, self.latent_size, 1, 1)
        self.conv_log_var = nn.Conv2d(out_channels, self.latent_size, 1, 1)

        self.conv_init_decoder = nn.Conv2d(self.latent_size, out_channels, 1, 1)

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
        # torchseg encoder yields list of tensors
        x_encoded = self.model.encoder(x)

        # encode the latent space
        mu = self.conv_mu(x_encoded[-1])
        log_var = self.conv_log_var(x_encoded[-1])
        z = self.reparameterization_trick(mu, log_var)

        # append the latent space to the encoder output
        x_encoded[-1] = self.conv_init_decoder(z)

        # decode
        x_decoded = self.model.decoder(*x_encoded)
        x_recon = self.model.segmentation_head(x_decoded)

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

        self.val_metrics(x_recon, y)

        return scaled_kl_loss + rec_loss

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
        x_encoded = self.model.encoder(X)

        # encode the latent space once
        mu = self.conv_mu(x_encoded[-1])
        log_var = self.conv_log_var(x_encoded[-1])

        # sample latent space and decode
        self.predictions: list[Tensor] = []
        for _ in range(self.num_samples):
            z = self.reparameterization_trick(mu, log_var)
            x_encoded[-1] = self.conv_init_decoder(z)
            x_decoded = self.model.decoder(*x_encoded)
            x_recon = self.model.segmentation_head(x_decoded)
            self.predictions.append(x_recon)

        preds = torch.stack(self.predictions, dim=-1)

        return {"pred": preds.mean(dim=-1), "pred_uct": preds.std(dim=-1)}


class ConditionalVAE(VAE):
    """Conditional Variational Auto Encoder for Torchseg."""

    def __init__(
        self,
        model: nn.Module,
        latent_size: int,
        loss_fn: nn.Module = VAELoss,
        num_samples: int = 5,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the Conditional VAE.

        Args:
            model: Torchseg model.
            latent_size: The size of the latent space.
            loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`.
            num_samples: The number of samples to draw from the latent space for prediction.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        super().__init__(
            model,
            loss_fn,
            freeze_backbone,
            freeze_decoder,
            optimizer,
            lr_scheduler,
            save_preds,
        )
