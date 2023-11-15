"""UQ-Regression-Box Models."""

from .cards import (
    ConditionalGuidedConvModel,
    ConditionalGuidedLinearModel,
    ConditionalLinear,
)
from .mlp import MLP

__all__ = (
    # custom models
    # CARDS
    "ConditionalLinear",
    "ConditionalGuidedLinearModel",
    "ConditionalGuidedConvModel",
    # Toy Example architecture
    "MLP",
)
