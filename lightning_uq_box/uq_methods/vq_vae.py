# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Vector Quantized Variational Auto Encoder (VQ-VAE)."""

import math
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid, save_image
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import gumbel_sample

from lightning_uq_box.models.pixel_cnn import PixelCNN
from lightning_uq_box.models.vae import VAEDecoder

from .base import BaseModule
from .loss_functions import VQVAELoss
from .vae import VAE


class VQVAE(VAE):
    """Vector Quantized Variational Auto Encoder (VQ-VAE).

    This is the first of the two stages described in
    `van den Oord et al. 2017 <https://arxiv.org/abs/1711.00937>`__: it learns a
    discrete latent representation of the input by quantizing the encoder output
    against a learned codebook, and reconstructs the input from it.

    The latent is a spatial grid of codebook indices of shape
    ``[B, latent_feature_dim, latent_feature_dim]``. Training an autoregressive prior
    over that grid, and thereby generating new images, is the job of
    :class:`VQVAEPrior`.

    This VQ-VAE is intended to be used with
    `SMP Encoders <https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/encoders/timm_universal.py>`__
    that support a wide range of `Timm Models <https://rwightman.github.io/pytorch-image-models/>`__, and
    `Lucidrains VQ Modules <https://github.com/lucidrains/vector-quantize-pytorch>`__.

    If you use this method in your work, please cite:

    * https://arxiv.org/abs/1711.00937
    """

    # Assigned in configure_model(), which always leaves it set. Declared here so it
    # reads as a Module rather than through nn.Module.__getattr__.
    vq_module: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        encoder: nn.Module,
        num_samples: int,
        out_channels: int,
        img_size: int,
        codebook_size: int = 512,
        sample_codebook_temp: float = 1.0,
        decoder_channels: list[int] | None = None,
        vq_module: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        log_samples_every_n_steps: int = 500,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        save_preds: bool = False,
        decay: float = 0.99,
        kmeans_init: bool = True,
        threshold_ema_dead_code: int = 2,
        rotation_trick: bool = True,
    ) -> None:
        """Initialize the VQ-VAE model.

        Args:
            encoder: Encoder Timm Model.
            num_samples: The number of samples to draw from the codebook for
                prediction. Values greater than one require
                ``sample_codebook_temp > 0`` to produce a non-zero uncertainty.
            out_channels: The number of output channels.
            img_size: The size of the input image, needed to configure the decoder by
                infering the output size of the encoder.
            codebook_size: The number of entries in the vector quantization codebook,
                called K in the paper, which uses 512 for CIFAR-10.
            sample_codebook_temp: The temperature for stochastic codebook sampling at
                prediction time **only**. Zero means deterministic nearest-neighbour
                lookup and therefore zero predictive uncertainty. Training and
                validation always use the deterministic nearest-neighbour assignment,
                regardless of this value; see :meth:`predict_step`.
            decoder_channels: The decoder channel sizes, excluding the output layer for
                the :class:`~.models.vae.VAEDecoder`, needs to match the encoder
                depth + 1. For example, with the standard resnet18 encoder, this would
                be [512, 256, 128, 64, 32, 16].
            vq_module: The VQ module to use, by default
                :class:`~vector_quantize_pytorch.VectorQuantize`. A custom module must
                be configured with ``accept_image_fmap=True`` and a ``dim`` matching the
                number of latent channels. See
                `Lucidrains VQ Modules <https://github.com/lucidrains/vector-quantize-pytorch>`__
                for the available options.
            loss_fn: The loss function, by default :class:`~.loss_functions.VQVAELoss`.
            freeze_backbone: Whether to freeze the backbone.
            freeze_decoder: Whether to freeze the decoder.
            log_samples_every_n_steps: How often to log reconstructions.
            optimizer: The optimizer to use.
            lr_scheduler: The learning rate scheduler.
            save_preds: Whether to save predictions.
            decay: The EMA decay for the codebook update. The library default of 0.8
                adapts too fast and destabilizes the codebook; 0.99 is the value used
                by the paper and by common reference implementations.
            kmeans_init: Whether to initialize the codebook from k-means centroids of
                the first batch. Random initialization leaves many codes far from the
                encoder output distribution, where they are never selected and stay
                dead.
            threshold_ema_dead_code: Codes whose EMA cluster size falls below this are
                resampled from the current batch. This is the standard dead-code
                revival mechanism; ``0`` disables it and lets usage decay
                monotonically.
            rotation_trick: Whether to propagate gradients through the quantizer with
                the rotation trick of `Fifty et al. 2024
                <https://arxiv.org/abs/2410.06424>`__ instead of the identity
                straight-through estimator of the original VQ-VAE paper. Both leave
                the forward pass exactly equal to the selected codebook entry and
                differ only in the backward pass. This is exposed explicitly because
                the underlying library silently enables it for any ``dim > 1``, which
                would otherwise make the gradient estimator an undocumented
                consequence of a library default rather than a stated choice. Set it
                to ``False`` to reproduce van den Oord et al. exactly.

        Raises:
            ValueError: If ``num_samples`` is greater than one but
                ``sample_codebook_temp`` is not positive, which would yield an
                identically zero predictive uncertainty.
        """
        if loss_fn is None:
            loss_fn = VQVAELoss()

        if num_samples > 1 and sample_codebook_temp <= 0:
            raise ValueError(
                "num_samples > 1 requires sample_codebook_temp > 0, otherwise every "
                "draw is the same deterministic nearest-neighbour lookup and "
                "'pred_uct' is identically zero. Got num_samples="
                f"{num_samples} and sample_codebook_temp={sample_codebook_temp}."
            )

        self.codebook_size = codebook_size
        self.sample_codebook_temp = sample_codebook_temp
        self._user_vq_module = vq_module
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.rotation_trick = rotation_trick

        # latent_size is Gaussian-VAE specific and unused here, but the VAE
        # signature requires it.
        super().__init__(
            encoder,
            0,
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
        """Configure all model parts.

        Raises:
            ValueError: If a user supplied ``vq_module`` is not compatible with the
                latent produced by the encoder.
        """
        # Lightning re-invokes configure_model at fit/validate/test whenever it is
        # overridden, which would otherwise rebuild these modules from scratch and
        # silently discard trained weights.
        if getattr(self, "_vqvae_model_configured", False):
            return

        if self.decoder_channels is None:
            # out_channels may be a tuple, and it is assigned to below
            self.decoder_channels = list(self.encoder.out_channels[::-1])
        self.decoder_channels[-1] = self.out_channels

        self.latent_channels = self.decoder_channels[0]
        self.latent_feature_dim = self.img_size // self.encoder.output_stride

        if self._user_vq_module is None:
            self.vq_module = VectorQuantize(
                dim=self.latent_channels,
                codebook_size=self.codebook_size,
                accept_image_fmap=True,
                # NOT stochastic_sample_codes=True. That flag is baked into a partial
                # at construction and gates on module.training, so enabling it here
                # would make *training* sample codes from Gumbel-perturbed distances
                # rather than take the nearest neighbour. The distance margin between
                # the best and second-best code is ~0.01 while the Gumbel noise has
                # std ~1.28, so the assignment becomes essentially uniform: measured,
                # only 0.3% of training assignments matched the true nearest
                # neighbour. predict_step rebinds this temporarily to get its
                # sampling uncertainty, which is the only place it belongs.
                stochastic_sample_codes=False,
                decay=self.decay,
                kmeans_init=self.kmeans_init,
                threshold_ema_dead_code=self.threshold_ema_dead_code,
                # Passed explicitly: the library turns this on by default whenever
                # dim > 1, which silently replaces the paper's identity
                # straight-through estimator. The forward pass is identical either
                # way, so no reconstruction or shape test can detect the difference
                # -- only the backward pass changes.
                rotation_trick=self.rotation_trick,
            )
        else:
            # Validate here rather than letting it fail deep inside the codebook
            # with an unhelpful shape error.
            if getattr(self._user_vq_module, "dim", None) != self.latent_channels:
                raise ValueError(
                    "The vq_module dim must match the number of latent channels, "
                    f"which is {self.latent_channels}, but got "
                    f"{getattr(self._user_vq_module, 'dim', None)}."
                )
            if not getattr(self._user_vq_module, "accept_image_fmap", False):
                raise ValueError(
                    "The vq_module must be configured with accept_image_fmap=True so "
                    "that the latent is quantized per spatial position, yielding a "
                    "grid of codebook indices."
                )
            self.vq_module = self._user_vq_module
            self.codebook_size = self._user_vq_module.codebook_size

        # Add segmentation head, because Decoder final layer is a ReLU
        self.decoder = VAEDecoder(decoder_channels=self.decoder_channels)

        self._vqvae_model_configured = True

    def freeze_model(self) -> None:
        """Freeze model backbone.

        Overrides :meth:`VAE.freeze_model`, which re-enables gradients on the
        Gaussian-specific ``latent_mu``/``latent_log_var`` modules that a VQ model
        does not have.
        """
        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def encode_img_to_latent(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode an image to the discrete latent space.

        Args:
            x: The input tensor of shape [B, C, H, W].

        Returns:
            The quantized latent of shape
            [B, latent_channels, latent_feature_dim, latent_feature_dim], the codebook
            indices of shape [B, latent_feature_dim, latent_feature_dim], and the
            scalar commitment loss.
        """
        x_enc = self.encoder_forward(x)
        quantized, indices, commit_loss = self.vq_module(x_enc)
        # With accept_image_fmap=True the codebook lookup permutes the channel axis
        # back into place, leaving a non-contiguous view. Downstream torchmetrics
        # calls .view(-1) on the reconstruction, which rejects such a tensor.
        return quantized.contiguous(), indices, commit_loss

    def forward(
        self, X: Tensor, cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the VQ-VAE.

        Args:
            X: The input tensor of shape [B, C, H, W].
            cond: The conditional tensor, unused.

        Returns:
            The reconstruction of shape [B, out_channels, img_size, img_size], the
            codebook indices, and the scalar commitment loss.
        """
        quantized, indices, commit_loss = self.encode_img_to_latent(X)
        x_recon = self.decoder(quantized)
        return x_recon, indices, commit_loss

    def compute_codebook_metrics(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Compute codebook usage diagnostics.

        A VQ-VAE whose codebook has collapsed onto a handful of entries is broken
        regardless of what the reconstruction loss says, so these are logged
        alongside the losses.

        Both quantities are computed **per batch**, so they are upper bounds on the
        codebook usage of the epoch: a code counts as used if any position in the
        batch selects it. They are also only meaningful in eval mode. In train mode
        the encoder's batch norm uses batch statistics, which shifts the latent and
        therefore the assignment, so train and validation values are not comparable
        with each other -- compare validation against validation.

        Args:
            indices: The codebook indices of shape [B, H, W].

        Returns:
            The perplexity, i.e. the effective number of codes in use, and the
            fraction of the codebook used in this batch.
        """
        counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
        probs = counts / counts.sum()
        # exp of the entropy, computed in a way that ignores the unused codes
        entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum()
        perplexity = entropy.exp()
        usage = (counts > 0).float().mean()
        return perplexity, usage

    def _shared_step(self, batch: Any, prefix: str) -> tuple[Tensor, Tensor]:
        """Run one training or validation step.

        Args:
            batch: The input batch.
            prefix: Either ``"train"`` or ``"val"``, used to prefix the logged metrics.

        Returns:
            The total loss and the reconstruction.
        """
        X, y = batch[self.input_key], batch[self.target_key]
        batch_size = X.shape[0]

        x_recon, indices, commit_loss = self.forward(X)
        scaled_commit_loss, rec_loss = self.loss_fn(x_recon, y, commit_loss)
        loss = scaled_commit_loss + rec_loss

        perplexity, usage = self.compute_codebook_metrics(indices)

        # NOTE: ``VectorQuantize`` only computes the commitment loss in training
        # mode and returns exactly 0.0 in eval, so ``val_commit_loss`` is always
        # 0.0 and ``val_loss`` equals ``val_rec_loss``. It is still logged to
        # keep the train/val metric sets symmetric, but do not read anything
        # into its value.
        self.log(f"{prefix}_commit_loss", scaled_commit_loss, batch_size=batch_size)
        self.log(f"{prefix}_rec_loss", rec_loss, batch_size=batch_size)
        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_codebook_perplexity", perplexity, batch_size=batch_size)
        self.log(f"{prefix}_codebook_usage", usage, batch_size=batch_size)

        return loss, x_recon

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Training step for the VQ-VAE.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The training loss.
        """
        loss, x_recon = self._shared_step(batch, "train")
        self.train_metrics(x_recon, batch[self.target_key])
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Validation step for the VQ-VAE.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The validation loss.
        """
        loss, x_recon = self._shared_step(batch, "val")
        self.val_metrics(x_recon, batch[self.target_key])

        if (
            self.trainer.global_step % self.log_samples_every_n_steps == 0
            and self.trainer.global_rank == 0
        ):
            self.plot_and_save_samples(batch)

        return loss

    @rank_zero_only
    def plot_and_save_samples(self, batch: dict[str, Tensor]) -> None:
        """Plot reconstructions from the VQ-VAE.

        Overrides :meth:`VAE.plot_and_save_samples`, which draws unconditional samples
        from the prior. A stage-one VQ-VAE has no prior to sample from, so
        reconstructions of the current batch are logged instead.

        Args:
            batch: The batch to reconstruct.
        """
        with torch.no_grad():
            X = batch[self.input_key]
            x_recon, _, _ = self.forward(X)

        # inputs on the top rows, their reconstructions below
        num_show = min(8, X.shape[0])
        paired = torch.cat([X[:num_show], x_recon[:num_show]], dim=0).detach()
        grid = make_grid(paired, nrow=num_show, normalize=True)
        save_image(
            grid,
            self.trainer.default_root_dir
            + f"/reconstruction_{self.trainer.global_step}.png",
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step with the VQ-VAE.

        Uncertainty comes from stochastic codebook sampling: instead of always taking
        the nearest codebook entry, each draw samples an entry with a Gumbel-softmax
        over the negative distances.

        Args:
            X: The input tensor of shape [B, C, H, W].
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            Prediction dictionary with the mean prediction and its standard deviation
            across draws.
        """
        with torch.no_grad():
            # the encoder is deterministic, so run it once
            x_enc = self.encoder_forward(X)

            codebook = self.vq_module._codebook
            was_training = codebook.training
            original_gumbel_sample = codebook.gumbel_sample
            try:
                # The codebook is built with stochastic_sample_codes=False so that
                # training uses the true nearest neighbour, so stochastic sampling is
                # rebound here for the duration of the prediction. It is also gated on
                # module.training inside gumbel_sample, hence the train() call: in eval
                # mode every draw would be identical and 'pred_uct' identically zero.
                # freeze_codebook keeps the EMA update from running while the codebook
                # is in train mode, which would otherwise make predictions depend on
                # the order of the test set.
                codebook.gumbel_sample = partial(
                    gumbel_sample, stochastic=True, straight_through=False
                )
                codebook.train()
                preds = [
                    self.decoder(
                        self.vq_module(
                            x_enc,
                            freeze_codebook=True,
                            sample_codebook_temp=self.sample_codebook_temp,
                        )[0].contiguous()
                    )
                    for _ in range(self.num_samples)
                ]
            finally:
                codebook.gumbel_sample = original_gumbel_sample
                codebook.train(was_training)

        stacked = torch.stack(preds, dim=-1)

        if self.num_samples == 1:
            # the unbiased std of a single sample is nan
            pred = stacked.squeeze(-1)
            return {"pred": pred, "pred_uct": torch.zeros_like(pred)}

        return {"pred": stacked.mean(dim=-1), "pred_uct": stacked.std(dim=-1)}

    def sample(self, num_samples: int = 16) -> Tensor:
        """Sampling is not available for a stage-one VQ-VAE.

        Args:
            num_samples: The number of samples to draw.

        Raises:
            NotImplementedError: Always. Generating new images requires a prior over
                the discrete latent, which is what :class:`VQVAEPrior` provides.
        """
        raise NotImplementedError(
            "A VQ-VAE has no prior over its discrete latent, so it cannot sample "
            "unconditionally. Train a VQVAEPrior on this model and call sample() on "
            "that instead."
        )


class VQVAEPrior(BaseModule):
    """Autoregressive PixelCNN prior over VQ-VAE codebook indices.

    This is the second of the two stages described in
    `van den Oord et al. 2017 <https://arxiv.org/abs/1711.00937>`__. It holds a
    trained, frozen :class:`VQVAE` and fits a :class:`~.models.pixel_cnn.PixelCNN`
    over the grid of codebook indices that VQ-VAE produces. Once fitted, new latent
    grids can be drawn ancestrally and decoded into images.

    If you use this method in your work, please cite:

    * https://arxiv.org/abs/1711.00937
    """

    # Declared so they read as their concrete types rather than through
    # nn.Module.__getattr__, which returns `Tensor | Module`.
    vq_vae: VQVAE
    pixel_cnn: PixelCNN

    def __init__(
        self,
        vq_vae: VQVAE,
        vq_vae_ckpt_path: str | None = None,
        c_hidden: int = 64,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize the VQ-VAE prior.

        Args:
            vq_vae: The stage-one VQ-VAE, whose encoder and codebook define the
                discrete latent this prior is fitted to. It is frozen and put in
                eval mode.
            vq_vae_ckpt_path: Optional path to a checkpoint whose weights are loaded
                into ``vq_vae``. The architecture given by ``vq_vae`` must match the
                checkpoint exactly.
            c_hidden: The number of hidden channels of the PixelCNN.
            optimizer: The optimizer to use. It only ever receives the PixelCNN
                parameters, since the VQ-VAE is frozen.
            lr_scheduler: The learning rate scheduler.
        """
        super().__init__()

        self.c_hidden = c_hidden
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.vq_vae = vq_vae

        if vq_vae_ckpt_path is not None:
            ckpt = torch.load(vq_vae_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            # strict, so that an architecture mismatch fails loudly here instead of
            # silently training a prior on a randomly initialized encoder
            self.vq_vae.load_state_dict(state_dict, strict=True)

        self.vq_vae.eval()
        for param in self.vq_vae.parameters():
            param.requires_grad = False

        # Built here rather than in configure_model: Lightning re-invokes
        # configure_model at fit, which would rebuild the PixelCNN and, worse,
        # re-run VQVAE.configure_model and discard the loaded weights.
        self.pixel_cnn = PixelCNN(
            num_embeddings=self.vq_vae.codebook_size, c_hidden=c_hidden
        )

    def setup_task(self) -> None:
        """Set up task specific attributes.

        The prior is a density model over discrete codes, so it logs cross entropy
        and accuracy directly rather than any of the regression metrics.
        """

    def train(self, mode: bool = True) -> "VQVAEPrior":
        """Set the module in training mode, keeping the frozen VQ-VAE in eval mode.

        Args:
            mode: Whether to set training mode (``True``) or evaluation mode.

        Returns:
            self
        """
        super().train(mode)
        self.vq_vae.eval()
        return self

    def forward(self, indices: Tensor) -> Tensor:
        """Predict codebook logits for a grid of indices.

        Args:
            indices: Codebook indices of shape [B, H, W].

        Returns:
            Logits of shape [B, codebook_size, H, W].
        """
        return self.pixel_cnn(indices)

    def _shared_step(self, batch: Any, prefix: str) -> Tensor:
        """Run one training or validation step.

        Args:
            batch: The input batch.
            prefix: Either ``"train"`` or ``"val"``, used to prefix the logged metrics.

        Returns:
            The cross entropy of the prior over the codebook indices, in nats.
        """
        X = batch[self.input_key]
        batch_size = X.shape[0]

        with torch.no_grad():
            _, indices, _ = self.vq_vae.encode_img_to_latent(X)

        logits = self.pixel_cnn(indices)
        loss = F.cross_entropy(logits, indices)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == indices).float().mean()

        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_acc", acc, batch_size=batch_size)
        # bits per latent code, which is the quantity the prior actually compresses
        self.log(f"{prefix}_bits_per_code", loss / math.log(2), batch_size=batch_size)

        return loss

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Training step for the prior.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Validation step for the prior.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The validation loss.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Test step for the prior.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            The test loss.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Any:
        """Initialize the optimizer and learning rate scheduler.

        Only the PixelCNN parameters are optimized, since the VQ-VAE is frozen.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer: Optimizer = self.optimizer(self.pixel_cnn.parameters())
        if self.lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler(optimizer),
                    "monitor": "val_loss",
                },
            }
        return {"optimizer": optimizer}

    def sample(self, num_samples: int = 16, temperature: float = 1.0) -> Tensor:
        """Generate new images by ancestral sampling of the discrete latent.

        The latent grid is filled one position at a time in raster-scan order. Each
        position is sampled from the PixelCNN's predictive distribution conditioned on
        all previously filled positions, and the completed grid of indices is mapped
        back to codebook vectors and decoded.

        This costs ``latent_feature_dim ** 2`` sequential forward passes.

        Args:
            num_samples: The number of images to generate.
            temperature: The softmax temperature. Values below one make samples more
                typical, values above one more diverse.

        Returns:
            The generated images of shape
            [num_samples, out_channels, img_size, img_size].
        """
        grid = self.vq_vae.latent_feature_dim

        was_training = self.pixel_cnn.training
        try:
            self.pixel_cnn.eval()
            with torch.no_grad():
                indices = torch.zeros(
                    num_samples, grid, grid, dtype=torch.long, device=self.device
                )
                for h in range(grid):
                    for w in range(grid):
                        logits = self.pixel_cnn(indices)[:, :, h, w] / temperature
                        probs = torch.softmax(logits, dim=-1)
                        # write back before the next position is predicted
                        indices[:, h, w] = torch.multinomial(probs, 1).squeeze(-1)

                # nn.Module.__getattr__ types submodule access as `Tensor | Module`,
                # so the concrete VectorQuantize method is not visible here.
                quantized = self.vq_vae.vq_module.get_output_from_indices(  # ty: ignore[call-non-callable]
                    indices
                )
                return self.vq_vae.decoder(quantized.contiguous())
        finally:
            self.pixel_cnn.train(was_training)
