# MIT License
# Copyright (c) 2020 Phillip Lippe
# Adapted from the UvA Deep Learning Tutorials:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""PixelCNN prior over VQ-VAE codebook indices.

Adapted from the original image-space PixelCNN to model a *discrete latent* instead:
the input is a grid of codebook indices produced by a
:class:`~lightning_uq_box.uq_methods.vq_vae.VQVAE`, and the output is a categorical
distribution over the codebook at every latent position. This is the autoregressive
prior of `van den Oord et al. 2017 <https://arxiv.org/abs/1711.00937>`__, which makes
ancestral sampling of new latents - and therefore of new images - possible.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskedConvolution(nn.Module):
    """Convolution with a fixed binary mask applied to its weights."""

    mask: Tensor

    def __init__(self, c_in: int, c_out: int, mask: Tensor, **kwargs) -> None:
        """Implement a convolution with a mask applied on its weights.

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            mask: Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                the convolution should be masked, and 1s otherwise.
            kwargs: Additional arguments for the convolution.
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = kwargs.get("dilation", 1)
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer("mask", mask[None, None])

    def forward(self, x: Tensor) -> Tensor:
        """Apply the masked convolution.

        Args:
            x: Input tensor of shape [B, c_in, H, W].

        Returns:
            Output tensor of shape [B, c_out, H, W].
        """
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):
    """Masked convolution for the vertical stack."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        mask_center: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the vertical stack convolution.

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            kernel_size: Size of the square convolution kernel.
            mask_center: Whether to additionally mask the center row, which is
                required for the very first layer so a position cannot see itself.
            kwargs: Additional arguments for the convolution.
        """
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):
    """Masked convolution for the horizontal stack."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        mask_center: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the horizontal stack convolution.

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            kernel_size: Size of the convolution kernel along the width.
            mask_center: Whether to additionally mask the center pixel, which is
                required for the very first layer so a position cannot see itself.
            kwargs: Additional arguments for the convolution.
        """
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):
    """Gated masked convolution block combining a vertical and horizontal stack."""

    def __init__(self, c_in: int, **kwargs) -> None:
        """Initialize the gated convolution block.

        Args:
            c_in: Number of input (and output) channels of the block.
            kwargs: Additional arguments for the masked convolutions, e.g. ``dilation``.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(
            2 * c_in, 2 * c_in, kernel_size=1, padding=0
        )
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack: Tensor, h_stack: Tensor) -> tuple[Tensor, Tensor]:
        """Run one gated masked convolution block.

        Args:
            v_stack: Vertical stack features of shape [B, c_in, H, W].
            h_stack: Horizontal stack features of shape [B, c_in, H, W].

        Returns:
            The updated vertical and horizontal stack features, each of
            shape [B, c_in, H, W].
        """
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class PixelCNN(nn.Module):
    """Gated PixelCNN over a grid of VQ-VAE codebook indices.

    Models ``p(indices)`` autoregressively in raster-scan order, so that a new
    latent grid can be drawn one position at a time and decoded into an image.
    """

    def __init__(self, num_embeddings: int, c_hidden: int = 64) -> None:
        """Initialize the PixelCNN prior.

        Args:
            num_embeddings: Size of the VQ-VAE codebook. This is both the number of
                distinct input symbols and the number of output classes.
            c_hidden: Number of hidden channels used throughout the gated stacks.
        """
        super().__init__()

        # Codebook indices are categorical, so embed them instead of rescaling
        # them like pixel intensities.
        self.embedding = nn.Embedding(num_embeddings, c_hidden)

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(
            c_hidden, c_hidden, mask_center=True
        )
        self.conv_hstack = HorizontalStackConvolution(
            c_hidden, c_hidden, mask_center=True
        )
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=4),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
            ]
        )
        # Output classification convolution (1x1) over the codebook
        self.conv_out = nn.Conv2d(c_hidden, num_embeddings, kernel_size=1, padding=0)

    def forward(self, indices: Tensor) -> Tensor:
        """Predict the codebook distribution at every latent position.

        Args:
            indices: Codebook indices of shape [B, H, W] with integer values in
                ``[0, num_embeddings)``.

        Returns:
            Logits of shape [B, num_embeddings, H, W], laid out so they can be passed
            directly to :func:`torch.nn.functional.cross_entropy` against ``indices``.
        """
        # [B, H, W] -> [B, c_hidden, H, W]
        x = self.embedding(indices).permute(0, 3, 1, 2)

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        return self.conv_out(F.elu(h_stack))
