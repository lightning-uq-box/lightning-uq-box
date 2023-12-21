# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""UQ-Regression-Box Models."""

from .cards import (
    ConditionalGuidedConvModel,
    ConditionalGuidedLinearModel,
    ConditionalLinear,
    DiffusionSequential,
)
from .hierarchical_prob_unet import (
    LagrangeMultiplier,
    MovingAverage,
    _HierarchicalCore,
    _StitchingDecoder,
)
from .mlp import MLP
from .prob_unet import AxisAlignedConvGaussian, Fcomb

__all__ = (
    # Toy Example architecture
    "MLP",
    # CARDS architecture
    "ConditionalLinear",
    "ConditionalGuidedLinearModel",
    "ConditionalGuidedConvModel",
    "DiffusionSequential",
    # Prob Unet
    "AxisAlignedConvGaussian",
    "Fcomb",
    # Hierarchical ProbUNet architecture
    "LagrangeMultiplier",
    "MovingAverage",
    "_HierarchicalCore",
    "_StitchingDecoder",
)
