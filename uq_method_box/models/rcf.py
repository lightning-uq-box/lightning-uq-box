"""Random Kitchen Sink Features Model with Prediction Head."""

from typing import Optional

import torch.nn as nn
from torch import Tensor
from torchgeo.models import RCF


class RCFLinearModel(RCF):
    """RCF feature extractor with Linear Prediction Head."""

    def __init__(
        self,
        prediction_head: nn.Module,
        in_channels: int = 4,
        features: int = 16,
        kernel_size: int = 3,
        bias: float = -1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize a new instance of RCF model."""
        super().__init__(in_channels, features, kernel_size, bias, seed)

        self.prediction_head = prediction_head

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = super().forward(x)
        return self.prediction_head(features)
