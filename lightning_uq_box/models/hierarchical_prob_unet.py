# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""SMP-backed hierarchical spatial latent U-Net (Kohl et al., 2019)."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import (
    UnetDecoder,
    UnetDecoderBlock,
)
from segmentation_models_pytorch.encoders import get_encoder
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from .prob_unet import init_weights_orthogonal_normal


class PreActResBlock(nn.Module):
    """Pre-activated residual block without normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: int | None = None,
        convs_per_block: int = 3,
    ) -> None:
        """Initialize projections and ReLU/3x3 convolutions at the inner width."""
        super().__init__()
        width = down_channels if down_channels is not None else out_channels
        if min(in_channels, out_channels, width, convs_per_block) < 1:
            raise ValueError("Channel counts and convs_per_block must be positive.")
        self.input_projection = (
            nn.Conv2d(in_channels, width, 1) if in_channels != width else nn.Identity()
        )
        self.output_projection = (
            nn.Conv2d(width, out_channels, 1)
            if width != out_channels
            else nn.Identity()
        )
        self.skip_projection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.convs = nn.Sequential(
            *[
                layer
                for _ in range(convs_per_block)
                for layer in (nn.ReLU(), nn.Conv2d(width, width, 3, padding=1))
            ]
        )
        self.apply(init_weights_orthogonal_normal)

    def forward(self, x: Tensor) -> Tensor:
        """Map [B, in_channels, H, W] to [B, out_channels, H, W]."""
        return self.skip_projection(x) + self.output_projection(
            self.convs(self.input_projection(x))
        )


class HierarchicalCoreOutput(NamedTuple):
    """Decoder features, encoder features, NHWC distributions and NCHW latents."""

    decoder_features: Tensor
    encoder_features: list[Tensor]
    distributions: list[Independent]
    used_latents: list[Tensor]


class HierarchicalUnetDecoder(UnetDecoder):
    """SMP decoder with autoregressive spatial Gaussians before upsampling."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        latent_dims: Sequence[int] = (1, 1, 1, 1),
        use_norm: str | bool = "batchnorm",
        interpolation_mode: str = "nearest",
        convs_per_block: int = 3,
        blocks_per_level: int = 1,
        posterior: bool = False,
    ) -> None:
        """Initialize stochastic stages and a deterministic stitching tail.

        The posterior stops at its last distribution: subsequent decoder blocks
        cannot affect any posterior latent and would be unused trainable weights.
        Additional residual blocks refine each stochastic feature before its head.
        """
        if not 0 < len(latent_dims) < n_blocks or any(d < 1 for d in latent_dims):
            raise ValueError(
                "latent_dims must be positive and leave at least one deterministic stitching stage."
            )
        if blocks_per_level < 1 or convs_per_block < 1:
            raise ValueError("blocks_per_level and convs_per_block must be positive.")
        super().__init__(
            encoder_channels,
            decoder_channels,
            n_blocks=n_blocks,
            use_norm=use_norm,
            interpolation_mode=interpolation_mode,
        )
        self.latent_dims = tuple(latent_dims)
        self.posterior = posterior
        # Channel arithmetic from segmentation_models_pytorch/decoders/unet/decoder.py.
        channels = list(encoder_channels[1:])[::-1]
        inputs = [channels[0], *decoder_channels[:-1]]
        skips = [*channels[1:], 0]
        self.latent_heads = nn.ModuleList()
        self.refinements = nn.ModuleList()
        for i, dim in enumerate(latent_dims):
            self.latent_heads.append(nn.Conv2d(inputs[i], 2 * dim, 1))
            self.refinements.append(
                nn.Sequential(
                    *[
                        PreActResBlock(
                            inputs[i],
                            inputs[i],
                            max(1, inputs[i] // 2),
                            convs_per_block,
                        )
                        for _ in range(blocks_per_level)
                    ]
                )
            )
            self.blocks[i] = UnetDecoderBlock(
                inputs[i] + dim,
                skips[i],
                decoder_channels[i],
                use_norm=use_norm,
                interpolation_mode=interpolation_mode,
            )
        if posterior:
            self.blocks = self.blocks[: len(latent_dims) - 1]
        self.latent_heads.apply(init_weights_orthogonal_normal)

    # SMP annotates forward as Tensor; this decoder also returns latent distributions.
    def forward(  # ty: ignore[invalid-method-override]
        self,
        features: list[Tensor],
        mean: bool | Sequence[bool] = False,
        z_q: Sequence[Tensor] | None = None,
    ) -> HierarchicalCoreOutput:
        """Decode encoder features, optionally forcing NCHW latents at every scale.

        Args:
            features: SMP encoder features, finest to coarsest.
            mean: Use Gaussian means globally or separately at each scale.
            z_q: Optional latent maps [B, latent_dims[i], H_i, W_i].

        Returns:
            Features, distributions with batch shape [B, H_i, W_i] and
            event shape [latent_dims[i]], and the exact latents used.
        """
        means = [mean] * len(self.latent_dims) if isinstance(mean, bool) else list(mean)
        if len(means) != len(self.latent_dims) or (
            z_q is not None and len(z_q) != len(means)
        ):
            raise ValueError("Provide one mean flag and one external latent per scale.")
        shapes = [f.shape[-2:] for f in features][::-1]
        reversed_features = features[1:][::-1]
        x = self.center(reversed_features[0])
        distributions: list[Independent] = []
        latents: list[Tensor] = []
        for i in range(max(len(self.blocks), len(self.latent_dims))):
            if i < len(self.latent_dims):
                x = self.refinements[i](x)
                mu, log_sigma = self.latent_heads[i](x).chunk(2, dim=1)
                # Channels form the event; spatial locations remain batch axes.
                # Bound log scales and compute distributions in FP32 to prevent
                # exp/variance overflow under mixed precision or early GECO updates.
                dist = Independent(
                    Normal(
                        mu.float().movedim(1, -1),
                        log_sigma.float().clamp(-10, 10).movedim(1, -1).exp(),
                    ),
                    1,
                )
                z = (
                    z_q[i]
                    if z_q is not None
                    else (dist.mean if means[i] else dist.rsample()).movedim(-1, 1)
                )
                if z.shape != mu.shape:
                    raise ValueError(
                        f"Latent {i} has shape {z.shape}; expected {mu.shape}."
                    )
                distributions.append(dist)
                latents.append(z)
                x = torch.cat([x, z], dim=1)
            if i < len(self.blocks):
                skip = (
                    reversed_features[i + 1] if i + 1 < len(reversed_features) else None
                )
                x = self.blocks[i](x, *shapes[i + 1], skip_connection=skip)
        return HierarchicalCoreOutput(x, features, distributions, latents)


class HierarchicalProbUNet(nn.Module):
    """Hierarchical Probabilistic U-Net with independent SMP prior/posterior encoders.

    This adapts the spatial latent hierarchy to SMP, rather than reproducing the
    paper's eight-scale residual encoder. Inputs are NCHW; outputs are logits.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 2,
        latent_dims: Sequence[int] = (1, 1, 1, 1),
        convs_per_block: int = 3,
        blocks_per_level: int = 1,
        decoder_use_norm: str | bool = "batchnorm",
        decoder_interpolation: str = "nearest",
    ) -> None:
        """Build two encoders, spatial latent decoders and the prior's output head.

        Args:
            encoder_name: SMP encoder name.
            encoder_depth: Number of downsampling stages.
            encoder_weights: Pretrained encoder weights or None.
            decoder_channels: Output widths, one per upsampling stage.
            in_channels: Image channels.
            classes: Segmentation logit channels (one for sigmoid binary output).
            latent_dims: Latent channels at each coarse-to-fine stochastic scale.
            convs_per_block: Convolutions in each pre-head residual block.
            blocks_per_level: Pre-head residual blocks at each stochastic scale.
            decoder_use_norm: SMP decoder normalization; False disables it.
            decoder_interpolation: SMP upsampling interpolation mode.
        """
        super().__init__()
        if classes < 1 or in_channels < 1:
            raise ValueError("classes and in_channels must be positive.")
        if (
            len(decoder_channels) != encoder_depth
            or not 0 < len(latent_dims) < encoder_depth
        ):
            raise ValueError(
                "Match decoder_channels to encoder_depth and leave a deterministic stitching stage."
            )
        self.classes = classes
        self.in_channels = in_channels
        self.latent_dims = tuple(latent_dims)
        self.prior_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.posterior_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels + classes,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.prior_decoder = HierarchicalUnetDecoder(
            self.prior_encoder.out_channels,
            decoder_channels,
            encoder_depth,
            latent_dims,
            decoder_use_norm,
            decoder_interpolation,
            convs_per_block,
            blocks_per_level,
        )
        self.posterior_decoder = HierarchicalUnetDecoder(
            self.posterior_encoder.out_channels,
            decoder_channels,
            encoder_depth,
            latent_dims,
            decoder_use_norm,
            decoder_interpolation,
            convs_per_block,
            blocks_per_level,
            posterior=True,
        )
        self.segmentation_head = SegmentationHead(
            decoder_channels[-1], classes, kernel_size=3
        )
        self.segmentation_head.apply(init_weights_orthogonal_normal)

    def sample(
        self,
        img: Tensor,
        mean: bool | Sequence[bool] = False,
        z_q: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """Generate [B, classes, H, W] logits from images and optional NCHW latents."""
        out = self.prior_decoder(self.prior_encoder(img), mean=mean, z_q=z_q)
        return self.segmentation_head(out.decoder_features)

    def reconstruct(
        self, img: Tensor, seg_one_hot: Tensor, mean: bool | Sequence[bool] = False
    ) -> tuple[Tensor, list[Independent], list[Independent]]:
        """Return posterior reconstruction logits and conditional q/p distributions.

        Args:
            img: Images [B, in_channels, H, W].
            seg_one_hot: Soft or one-hot targets [B, classes, H, W].
            mean: Use posterior means globally or per scale.
        """
        q = self.posterior_decoder(
            self.posterior_encoder(torch.cat([img, seg_one_hot], dim=1)), mean=mean
        )
        p = self.prior_decoder(self.prior_encoder(img), z_q=q.used_latents)
        # Reconstruction uses PRIOR features conditioned on posterior samples.
        return (
            self.segmentation_head(p.decoder_features),
            q.distributions,
            p.distributions,
        )

    def forward(
        self, img: Tensor, seg_one_hot: Tensor, mean: bool | Sequence[bool] = False
    ) -> tuple[Tensor, list[Independent], list[Independent]]:
        """Run posterior and teacher-forced prior; omit the unused free-prior pass."""
        return self.reconstruct(img, seg_one_hot, mean)
