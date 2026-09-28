# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Threshold network shared by adaptive conformal segmentation methods."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ThresholdPredictor(nn.Module):
    """Predict a sigmoid threshold from an image and foreground probabilities."""

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        """Initialize a ResNet-50 whose stem takes the image plus one probability map.

        Image channel ``i`` is initialized from stem channel ``i % 3`` scaled by
        ``3 / in_channels``, so the summed response to a constant input matches the
        RGB stem (timm's ``adapt_input_conv`` convention). With ``in_channels=3`` the
        image channels are the unmodified RGB stem. The probability channel starts
        from the mean over the RGB channels.

        Args:
            pretrained: whether to load ImageNet weights, which may download
            in_channels: number of image channels, e.g. 6 for a stacked image pair

        Raises:
            ValueError: if in_channels is not positive
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        self.in_channels = in_channels
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        original = self.resnet.conv1.weight
        widened = nn.Conv2d(
            in_channels + 1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            rgb = original[:, torch.arange(in_channels) % 3]
            widened.weight[:, :in_channels].copy_(rgb * (3 / in_channels))
            widened.weight[:, in_channels:].copy_(original.mean(dim=1, keepdim=True))
        self.resnet.conv1 = widened
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, image: Tensor, phat: Tensor) -> Tensor:
        """Predict one threshold per image.

        Args:
            image: images of shape [batch_size x in_channels x H x W]
            phat: foreground probabilities of shape [batch_size x 1 x h x w] or
                [batch_size x h x w], resized bilinearly to H x W if needed

        Returns:
            thresholds in [0, 1] of shape [batch_size]

        Raises:
            ValueError: if the image or probability shapes do not match
        """
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected images with shape [B,{self.in_channels},H,W], "
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
