"""UQ-Regression-Box Models."""

from .bnnlv.latent_variable_network import LatentVariableNetwork
from .mlp import MLP

__all__ = (
    # custom models
    "MLP",
    # Latent Variable Network
    "LatentVariableNetwork",
)
