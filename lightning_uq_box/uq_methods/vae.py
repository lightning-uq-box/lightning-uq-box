# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Variational Auto Encoder (VAE)."""

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from torch.optim.adam import Adam as Adam
from torchvision.utils import make_grid, save_image

from lightning_uq_box.models.vae import VAEDecoder

from .base import DeterministicPixelRegression
from .loss_functions import VAELoss
from .utils import _get_num_inputs


class VAE(DeterministicPixelRegression):
    """Variational Auto Encoder (VAE) for Torchseg Encoders.

    This VAE is intended to be used with
    `Torchseg Encoders <https://github.com/isaaccorley/torchseg/blob/main/torchseg/encoders/timm.py>`__
    that support a wide range og `Timm Models <https://rwightman.github.io/pytorch-image-models/>`__.

    If you use this method in your work, please cite:

    * https://arxiv.org/abs/1312.6114
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_size: int,
        num_samples: int,
        out_channels: int,
        img_size: int,
        decoder_channels: list[int] | None = None,
        loss_fn: nn.Module | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        log_samples_every_n_steps: int = 500,
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
            img_size: The size of the input image, needed to configure the decoder by infering
                the output size of the encoder.
            decoder_channels: The decoder channel sizes, excluding the output layer for the
                :calss:`~.models.vae.VAEDecoder`., needs to match the encoder depth + 1. For example,
                with the standard resnet18 encoder, this would be [512, 256, 128, 64, 32, 16].
            loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`. The kl_scale
                factor in that loss function can have a significant impact on the performance of the VAE.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            log_samples_every_n_steps: How often to log samples.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        if loss_fn is None:
            loss_fn = VAELoss()

        self.out_channels = out_channels
        self.decoder_channels = decoder_channels
        self.latent_size = latent_size
        self.num_samples = num_samples
        self.img_size = img_size
        self.log_samples_every_n_steps = log_samples_every_n_steps

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
        self.configure_model()

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
            for param in self.latent_mu.parameters():
                param.requires_grad = True
            for param in self.latent_log_var.parameters():
                param.requires_grad = True

        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.latent_init_decoder.parameters():
                param.requires_grad = False

    def configure_model(self) -> None:
        """Configure all model parts."""
        if self.decoder_channels is None:
            self.decoder_channels = self.encoder.out_channels[::-1]
        self.decoder_channels[-1] = self.out_channels

        self.latent_channels = self.decoder_channels[0]

        self.latent_feature_dim = self.img_size // self.encoder.reductions[-1]

        # TODO: change this to be sequential to a linear latent space
        self.latent_mu = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.latent_size, 1, 1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                self.latent_size * self.latent_feature_dim * self.latent_feature_dim,
                self.latent_size,
            ),
        )
        self.latent_log_var = nn.Sequential(
            nn.Conv2d(self.latent_channels, self.latent_size, 1, 1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                self.latent_size * self.latent_feature_dim * self.latent_feature_dim,
                self.latent_size,
            ),
        )

        self.latent_init_decoder = nn.Sequential(
            nn.Linear(
                self.latent_size,
                self.latent_size * self.latent_feature_dim * self.latent_feature_dim,
            ),
            nn.Unflatten(
                1, (self.latent_size, self.latent_feature_dim, self.latent_feature_dim)
            ),
            nn.Conv2d(self.latent_size, self.latent_channels, 1, 1),
        )

        # Add segmentation head, because Decoder final layer is a ReLU
        self.decoder = VAEDecoder(decoder_channels=self.decoder_channels)

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for metrics."""
        return out

    def reparameterization_trick(self, mu: Tensor, log_var: Tensor) -> Tensor:
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

    def encoder_forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Encoder Forward pass.

        Args:
            x: The input image tensor.
            cond: The conditional tensor.

        Returns:
            The encoded image tensor.
        """
        return self.encoder.forward(x)[-1]

    def decoder_forward(self, z: Tensor) -> Tensor:
        """Decoder Forward pass.

        Args:
            z: The latent tensor

        Returns:
            The decoded tensor.
        """
        return self.decoder(self.latent_init_decoder(z))

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Forward pass for the VAE.

        Args:
            x: The input tensor
            cond: The cond tensor.

        Returns:
            The output tensor.
        """
        x_encoded = self.encoder_forward(x, cond)

        # encode the latent space
        mu = self.latent_mu(x_encoded)
        log_var = self.latent_log_var(x_encoded)
        z = self.reparameterization_trick(mu, log_var)

        # decode
        x_recon = self.decoder_forward(z)

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

        x_recon, mu, log_var = self.forward(
            X, cond=batch["condition"] if "condition" in batch else None
        )

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

        x_recon, mu, log_var = self.forward(
            X, cond=batch["condition"] if "condition" in batch else None
        )

        scaled_kl_loss, rec_loss = self.loss_fn(x_recon, y, mu, log_var)

        self.log("val_kl_loss", scaled_kl_loss, batch_size=batch_size)
        self.log("val_rec_loss", rec_loss, batch_size=batch_size)
        self.log("val_loss", scaled_kl_loss + rec_loss, batch_size=batch_size)

        self.val_metrics(x_recon, y)

        if (
            self.trainer.global_step % self.log_samples_every_n_steps == 0
            and self.trainer.global_rank == 0
        ):
            # log samples
            self.plot_and_save_samples(batch)

        return scaled_kl_loss + rec_loss

    @rank_zero_only
    def plot_and_save_samples(self, batch: dict[str, Tensor]) -> None:
        """Plot samples from VAE model."""
        with torch.no_grad():
            sampled_imgs = self.sample(num_samples=16).detach()

        grid = make_grid(sampled_imgs, nrow=4, normalize=True)
        save_image(
            grid,
            self.trainer.default_root_dir + f"/sample_{self.trainer.global_step}.png",
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction Step with VAE.

        Args:
            X: The input tensor.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            Prediction dictionary
        """
        x_encoded = self.encoder_forward(X, cond=None)

        # encode the latent space once
        mu = self.latent_mu(x_encoded)
        log_var = self.latent_log_var(x_encoded)

        # sample latent space and decode
        self.predictions: list[Tensor] = []
        for _ in range(self.num_samples):
            z = self.reparameterization_trick(mu, log_var)
            x = self.decoder_forward(z)
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
        z = torch.randn(num_samples, self.latent_size).to(self.device)
        x_decoded = self.decoder_forward(z)
        return x_decoded


class ConditionalVAE(VAE):
    """Conditional Variational Auto Encoder for Torchseg."""

    def __init__(
        self,
        encoder: nn.Module,
        latent_size: int,
        num_samples: int,
        out_channels: int,
        img_size: int,
        num_conditions: int,
        decoder_channels: list[int] | None = None,
        loss_fn: nn.Module | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        log_samples_every_n_steps: int = 500,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the Conditional VAE.

        Args:
            encoder: Torchseg model.
            latent_size: The size of the latent space.
            num_samples: The number of samples to draw from the latent space for prediction.
            out_channels: The number of output channels.
            img_size: The size of the input image, needed to configure the decoder and
                sampling procedure
            num_conditions: The number of discrete conditions, for example
                class labels (in the case of MNIST or EuroSAT this would be 10)
            decoder_channels: The decoder channel sizes, excluding the output layer for the
                :calss:`~.models.vae.VAEDecoder`., needs to match the encoder depth + 1. For example,
                with the standard resnet18 encoder, this would be [512, 256, 128, 64, 32, 16].
            loss_fn: The loss function, by default :class:`~.loss_functions.VAELoss`.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            log_samples_every_n_steps: How often to log samples.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        self.num_conditions = num_conditions

        super().__init__(
            encoder,
            latent_size,
            num_samples,
            out_channels,
            img_size,
            decoder_channels,
            loss_fn,
            freeze_backbone,
            freeze_decoder,
            log_samples_every_n_steps,
            optimizer,
            lr_scheduler,
            save_preds,
        )

    def configure_model(self) -> None:
        """Configure all model parts."""
        super().configure_model()

        # Additionally also define class embedding
        self.embed_dim = 1
        self.cond_embed = nn.Sequential(
            nn.Linear(1, self.num_conditions),
            nn.ReLU(),
            nn.Linear(self.num_conditions, self.embed_dim),
        )

        # init_decoder takes in latent space plus class conditional
        self.latent_init_decoder = nn.Sequential(
            nn.Linear(
                self.latent_size + self.embed_dim,
                self.latent_size * self.latent_feature_dim * self.latent_feature_dim,
            ),
            nn.Unflatten(
                1, (self.latent_size, self.latent_feature_dim, self.latent_feature_dim)
            ),
            nn.Conv2d(self.latent_size, self.latent_channels, 1, 1),
        )

    def encoder_forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Encode forward pass.

        Args:
            x: input tensor
            cond: cond tensor

        Returns:c
            encoded tensor
        """
        batch_size = x.shape[0]

        embed_condition = self.cond_embed(cond)

        # reshape to image space
        embed_cond = torch.ones_like(x) * embed_condition.view(
            batch_size, self.embed_dim, 1, 1
        )

        x = torch.cat([x, embed_cond], dim=1)

        return self.encoder.forward(x)[-1]

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Forward pass for the VAE.

        Args:
            x: The input tensor
            cond: The cond tensor.

        Returns:
            The output tensor.
        """
        x_encoded = self.encoder_forward(x, cond)

        # encode the latent space
        mu = self.latent_mu(x_encoded)
        log_var = self.latent_log_var(x_encoded)
        z = self.reparameterization_trick(mu, log_var)

        # init_decoder takes in latent space plus class conditional
        x_decoder_init = self.latent_init_decoder(torch.cat([z, cond.float()], dim=1))

        # decode
        x_recon = self.decoder(x_decoder_init)

        return x_recon, mu, log_var

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dictionary
        """
        out_dict = self.predict_step(
            batch[self.input_key],
            cond=batch["condition"] if "condition" in batch else None,
        )
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(out_dict["pred"]), batch[self.target_key]
            )

        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)

        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        if "out" in out_dict:
            del out_dict["out"]

        return out_dict

    def predict_step(
        self, X: Tensor, cond: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction Step with Conditional VAE.

        Args:
            X: The input tensor.
            cond: The condition tensor.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            Prediction dictionary
        """
        x_encoded = self.encoder_forward(X, cond)

        # encode the latent space once
        mu = self.latent_mu(x_encoded)
        log_var = self.latent_log_var(x_encoded)

        # sample latent space and decode
        self.predictions: list[Tensor] = []
        for _ in range(self.num_samples):
            z = self.reparameterization_trick(mu, log_var)
            x = self.latent_init_decoder(torch.cat([z, cond.float()], dim=1))
            x = self.decoder(x)
            self.predictions.append(x)

        preds = torch.stack(self.predictions, dim=-1)

        return {"pred": preds.mean(dim=-1), "pred_uct": preds.std(dim=-1)}

    def sample(self, num_samples: int = 16, cond: Tensor | None = None) -> Tensor:
        """Sample with conditioning from the VAE.

        Args:
            num_samples: number of samples
            cond: conditioning tensor

        Returns:
            samples
        """
        z = torch.randn(num_samples, self.latent_size).to(self.device)
        if cond is None:
            cond = torch.randint(0, self.num_conditions, (num_samples, 1)).to(
                self.device
            )
        x_decoded = self.decoder(self.latent_init_decoder(torch.cat([z, cond], dim=1)))
        return x_decoded
