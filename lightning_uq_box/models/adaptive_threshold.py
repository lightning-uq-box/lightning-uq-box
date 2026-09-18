# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Threshold network shared by adaptive conformal segmentation methods."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ThresholdPredictor(nn.Module):
    """Predict a sigmoid threshold from an RGB image and foreground probabilities."""

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize ResNet-50, optionally using ImageNet weights."""
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        original = self.resnet.conv1
        widened = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            widened.weight[:, :3].copy_(original.weight)
            widened.weight[:, 3:4].copy_(original.weight.mean(dim=1, keepdim=True))
        self.resnet.conv1 = widened
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, image: Tensor, phat: Tensor) -> Tensor:
        """Map image [B,3,H,W] and probabilities [B,1,h,w] or [B,h,w] to [B]."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB images with shape [B,3,H,W].")
        if phat.ndim == 3:
            phat = phat.unsqueeze(1)
        if phat.ndim != 4 or phat.shape[:2] != (image.shape[0], 1):
            raise ValueError("Expected binary probabilities with shape [B,1,H,W].")
        if phat.shape[-2:] != image.shape[-2:]:
            phat = F.interpolate(
                phat, image.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.resnet(torch.cat((image, phat), dim=1)).sigmoid().flatten()
