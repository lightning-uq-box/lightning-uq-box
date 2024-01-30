# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""UQ-Regression-Box Models."""

from .cards import (
    ConditionalEncoder,
    ConditionalGuidedConvModel,
    ConditionalGuidedLinearModel,
    ConditionalLinear,
    DiffusionSequential,
)
from .mlp import MLP

__all__ = (
    # Toy Example architecture
    "MLP",
    # CARDS architecture
    "ConditionalLinear",
    "ConditionalEncoder",
    "ConditionalGuidedLinearModel",
    "ConditionalGuidedConvModel",
    "DiffusionSequential",
)
