# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.adam import Adam as Adam
from vector_quantize_pytorch import VectorQuantize

from lightning_uq_box.models.vae import VAEDecoder

from .vae import VAE


class VQVAE(VAE):
    """Vector Quantized VAE.

    This VQ-VAE model is intended to be use with
    `Torchseg Encoders <https://github.com/isaaccorley/torchseg/blob/main/torchseg/encoders/timm.py>`__
    that support a wide range og `Timm Models <https://rwightman.github.io/pytorch-image-models/>`__, and
    `Lucidrains VQ Modules <https://github.com/lucidrains/vector-quantize-pytorch>`__.
    """

    def __init__(
        self,
        encoder: Module,
        out_channels: int,
        img_size: int,
        num_samples: int = 1,
        decoder_channels: list[int] | None = None,
        vq_module: nn.Module | None = None,
        loss_fn: Module | None = nn.MSELoss(),
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        pixel_cnn: nn.Module | None = None,
        log_samples_every_n_steps: int = 500,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
        save_preds: bool = False,
    ) -> None:
        """Initialize the VQ-VAE model.

        Args:
            encoder: Encoder Timm Model
            num_samples: The number of samples to draw from the latent space for prediction.
            out_channels: The number of output channels.
            img_size: The size of the input image, needed to configure the decoder by infering
                the output size of the encoder.
            decoder_channels: The decoder channel sizes, excluding the output layer for the
                :calss:`~.models.vae.VAEDecoder`., needs to match the encoder depth + 1. For example,
                with the standard resnet18 encoder, this would be [512, 256, 128, 64, 32, 16].
            loss_fn: The loss function, by default :class:`~torch.nn.MSELoss`, which computes the
                reconstruction loss between input and output. The VQ loss is computed internally, see
                :meth:`~VQVAE.training_step`.
            vq_module: The VQ module to use, by default :class:`~vector_quantize_pytorch.VectorQuantize`.
                See `Lucidrains VQ Modules <https://github.com/lucidrains/vector-quantize-pytorch>`__ for
                more available options.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            log_samples_every_n_steps: How often to log samples.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
        """
        super().__init__(
            encoder,
            0,  # overwrite latent size because using VQ
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

        if vq_module is None:
            self.vq_module = VectorQuantize(
                dim=self.latent_feature_dim**2,
                codebook_size=256,
                codebook_dim=16,
                use_cosine_sim=True,
            )
        else:
            assert vq_module.dim == self.latent_feature_dim**2, (
                f"The VQ module dim must match the latent feature dim squared, which is {self.latent_feature_dim**2}.",
                "The latent feature dim can be computed as img_size / encoder.reductions[-1].",
            )
            self.vq_module = vq_module

        if pixel_cnn is not None:
            assert (
                pixel_cnn.c_in == self.latent_channels
            ), f"The PixelCNN input channels must match the latent channels, which is {self.latent_channels}."
            self.pixel_cnn = pixel_cnn
            self.train_pixel_cnn = True
            if self.loss_fn is None:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.pixel_cnn = None
            self.train_pixel_cnn = False

    def configure_model(self) -> None:
        """Configure all model parts."""
        if self.decoder_channels is None:
            self.decoder_channels = self.encoder.out_channels[::-1]
        self.decoder_channels[-1] = self.out_channels

        self.latent_channels = self.decoder_channels[0]

        self.latent_feature_dim = self.img_size // self.encoder.reductions[-1]

        self.decoder = VAEDecoder(decoder_channels=self.decoder_channels)

    def encode_img_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image to the latent space.

        Args:
            x: The input tensor.

        Returns:
            The latent tensor.
        """
        x_enc = self.encoder_forward(x)
        import pdb

        pdb.set_trace()

        x_enc = rearrange(x_enc, "b c h w -> b c (h w)")

        # obtain their discrete representation as indices in the latent codebook
        quantized, embed_ind, vq_loss = self.vq_module(x_enc)

        quantized = rearrange(
            quantized, "b c (h w) -> b c h w", h=self.latent_feature_dim
        )

        return quantized, vq_loss

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of VQ-VAE.

        Args:
            x: The input tensor.
            cond: The conditional tensor.

        Returns:
            The output tensor.
        """
        quantized, vq_loss = self.encode_img_to_latent(x)

        x_dec = self.decoder(quantized)

        return x_dec, vq_loss, quantized

    def vq_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        eval: bool = False,
    ) -> torch.Tensor:
        """Training logic for VQ-VAE.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.
            eval: Whether to run in evaluation mode.

        Returns:
            The loss.
        """
        X, y = batch[self.input_key], batch[self.target_key]
        batch_size = X.shape[0]

        x_recon, vq_loss, _ = self.forward(X)

        rec_loss = self.loss_fn(x_recon, y)

        loss = rec_loss + vq_loss

        prefix = "val" if eval else "train"

        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_rec_loss", rec_loss, batch_size=batch_size)
        self.log(f"{prefix}_vq_loss", vq_loss, batch_size=batch_size)

        return loss, x_recon

    def pixel_cnn_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training logic for PixelCNN."""
        X, y = batch[self.input_key], batch[self.target_key]
        batch_size = X.shape[0]

        prefix = "val" if eval else "train"

        quantized, _ = self.encode_img_to_latent(X)

        import pdb

        pdb.set_trace()

        px_logits = self.pixel_cnn(quantized)

        # Here the training objective is a multi-class classification one, as the prior should output a map of codebook indices.
        # https://www.kaggle.com/code/ameroyer/keras-vq-vae-for-image-generation#Training-and-Testing-the-prior

        # https://github.com/rosinality/vq-vae-2-pytorch

        loss = self.loss_fn(px_logits, y)

        return loss, px_logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for VQ-VAE.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.

        Returns:
            The loss.
        """
        if self.train_pixel_cnn:
            loss, x_recon = self.pixel_cnn_step(batch, batch_idx, eval=False)
        else:
            loss, x_recon = self.vq_step(batch, batch_idx, eval=False)

        self.train_metrics(x_recon, batch[self.target_key])

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for VQ-VAE.

        Args:
            batch: The batch of data.
            batch_idx: The batch index.

        Returns:
            The loss.
        """
        if self.train_pixel_cnn:
            loss, x_recon = self.pixel_cnn_step(batch, batch_idx, eval=True)
        else:
            loss, x_recon = self.vq_step(batch, batch_idx, eval=True)

        self.val_metrics(x_recon, batch[self.target_key])

        return loss

    def predict_step(
        self, X: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step for VQ-VAE.

        Args:
            X: The input tensor.
            batch_idx: The batch index.
            dataloader_idx: The dataloader index.

        Returns:
            The prediction dictionary.
        """
        x_enc = self.encoder_forward(X)

        x_enc = rearrange(x_enc, "b c h w -> b c (h w)")

        if self.num_samples > 1:
            assert (
                self.vq_module.stochastic_sample_codes == True
            ), "The VQ module must have `stochastic_sample_codes` set to True, to sample multiple times."
        samples = []
        for i in range(self.num_samples):
            quantized, _, _ = self.vq_module(x_enc)

            quantized = rearrange(
                quantized, "b c (h w) -> b c h w", h=self.latent_feature_dim
            )
            samples.append(self.decoder(quantized))

        preds = torch.stack(samples, dim=-1)

        return {"pred": preds.mean(dim=-1), "pred_uct": preds.std(dim=-1)}

    def sample(self, num_samples: int = 16) -> Tensor:
        """Sample from the VQ-VAE.

        Args:
            num_samples: The number of samples to draw.

        Returns:
            The samples.

        Raises:
            Warning: If PixelCNN is not supplied.
        """
        if self.pixel_cnn is None:
            raise Warning("Sampling for VQ VAE is not implemented without PixelCNN.")
        else:
            pass

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        if self.train_pixel_cnn:
            optimizer = self.optimizer(self.pixel_cnn.parameters())
        else:
            optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
