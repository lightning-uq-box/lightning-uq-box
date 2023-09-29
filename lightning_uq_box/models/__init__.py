"""UQ-Regression-Box Models."""

from .mlp import MLP
from .cards import ConditionalLinear, ConditionalGuidedLinearModel, ConditionalGuidedConvModel

__all__ = (
    # custom models
    # CARDS
    "ConditionalLinear",
    "ConditionalGuidedLinearModel",
    "ConditionalGuidedConvModel",
    # Toy Example architecture
    "MLP",
)
