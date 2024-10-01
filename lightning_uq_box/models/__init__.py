# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""UQ-Regression-Box Models."""

from .cards import (
    ConditionalGuidedConvModel,
    ConditionalGuidedLinearModel,
    ConditionalLinear,
    DiffusionSequential,
)
from .density_layers import DensityConv2d, DensityLinear
from .fc_resnet import FCResNet
from .mixture_density import MixtureDensityLayer
from .mlp import MLP

__all__ = (
    # Toy Example architecture
    "MLP",
    # Mixture Density Layer
    "MixtureDensityLayer",
    # CARDS architecture
    "ConditionalLinear",
    "ConditionalGuidedLinearModel",
    "ConditionalGuidedConvModel",
    "DiffusionSequential",
    # Density Layers Uncertainty
    "DensityLinear",
    "DensityConv2d",
    # Fully Connected Residual Network
    "FCResNet",
)
