# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Threshold network shared by adaptive conformal segmentation methods."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _adapt_stem(rgb: Tensor, in_channels: int) -> Tensor:
    """Widen RGB stem kernels [64,3,k,k] to ``in_channels`` image channels plus phat.

    Image channel ``i`` copies RGB channel ``i % 3`` scaled by ``3 / in_channels``, so
    the summed response to a constant input matches the RGB stem (timm's
    ``adapt_input_conv`` convention). The probability channel starts from the RGB mean.
    """
    image = rgb[:, torch.arange(in_channels) % 3] * (3 / in_channels)
    return torch.cat((image, rgb.mean(dim=1, keepdim=True)), dim=1)


class _LazyStem(nn.LazyConv2d):
    """ResNet stem whose input width is read from the first batch.

    The weight is materialized in place, so an optimizer built before the first batch
    still updates it, and ``load_state_dict`` sizes it from a checkpoint.
    """

    rgb: Tensor

    def __init__(self, rgb: Tensor) -> None:
        """Keep the RGB kernels to adapt once the input width is known."""
        super().__init__(64, kernel_size=7, stride=2, padding=3, bias=False)
        self.register_buffer("rgb", rgb.detach().clone(), persistent=False)

    def reset_parameters(self) -> None:
        """Initialize from the RGB kernels once materialized."""
        # LazyConv2d.__init__ calls this before the weight is materialized or the
        # buffer exists; initialize_parameters calls it again after materializing.
        uninitialized = isinstance(self.weight, nn.parameter.UninitializedParameter)
        if uninitialized or "rgb" not in self._buffers:
            return
        with torch.no_grad():
            self.weight.copy_(_adapt_stem(self.rgb, self.weight.shape[1] - 1))


class ThresholdPredictor(nn.Module):
    """Predict a sigmoid threshold from an image and foreground probabilities."""

    def __init__(self, pretrained: bool = True, in_channels: int | None = None) -> None:
        """Initialize a ResNet-50 whose stem takes the image plus one probability map.

        By default the number of image channels is read from the first batch, so the
        same network serves RGB, multispectral or stacked bitemporal inputs. Image
        channel ``i`` is initialized from stem channel ``i % 3`` scaled by
        ``3 / in_channels``; with three channels this is the unmodified RGB stem. The
        probability channel starts from the mean over the RGB channels.

        Args:
            pretrained: whether to load ImageNet weights, which may download
            in_channels: number of image channels; ``None`` infers it from the first
                batch. Set it when the parameters must exist before any data is seen,
                e.g. for DistributedDataParallel.

        Raises:
            ValueError: if in_channels is not positive
        """
        super().__init__()
        if in_channels is not None and in_channels < 1:
            raise ValueError("in_channels must be positive.")
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        rgb = self.resnet.conv1.weight.detach()
        if in_channels is None:
            self.resnet.conv1 = _LazyStem(rgb)
        else:
            stem = nn.Conv2d(
                in_channels + 1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                stem.weight.copy_(_adapt_stem(rgb, in_channels))
            self.resnet.conv1 = stem
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    @property
    def in_channels(self) -> int | None:
        """Number of image channels, or ``None`` before the first batch."""
        weight = self.resnet.conv1.weight
        if isinstance(weight, nn.parameter.UninitializedParameter):
            return None
        return weight.shape[1] - 1

    def forward(self, image: Tensor, phat: Tensor) -> Tensor:
        """Predict one threshold per image.

        Args:
            image: images of shape [batch_size x C x H x W]
            phat: foreground probabilities of shape [batch_size x 1 x h x w] or
                [batch_size x h x w], resized bilinearly to H x W if needed

        Returns:
            thresholds in [0, 1] of shape [batch_size]

        Raises:
            ValueError: if the image or probability shapes do not match
        """
        expected = self.in_channels
        if image.ndim != 4 or (expected is not None and image.shape[1] != expected):
            raise ValueError(
                f"Expected images with shape [B,{expected or 'C'},H,W], "
                f"got {tuple(image.shape)}."
            )
        if phat.ndim == 3:
            phat = phat.unsqueeze(1)
        if phat.ndim != 4 or phat.shape[:2] != (image.shape[0], 1):
            raise ValueError("Expected binary probabilities with shape [B,1,H,W].")
        if phat.shape[-2:] != image.shape[-2:]:
            phat = F.interpolate(
                phat, image.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.resnet(torch.cat((image, phat), dim=1)).sigmoid().flatten()
