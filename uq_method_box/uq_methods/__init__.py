"""UQ-Methods as Lightning Modules."""

from .base import BaseModel
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .deterministic_gaussian import DeterministicGaussianModel
from .laplace_model import LaplaceModel
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel
from .sgld_nll import SGLDModel

__all__ = (
    # base model
    "BaseModel",
    # conformalized Quantile Regression
    "CQR",
    # MC-Dropout
    "MCDropoutModel",
    # Laplace Approximation
    "LaplaceModel",
    # Quantile Regression
    "QuantileRegressionModel",
    # Deep Ensemble Wrapper
    "DeepEnsembleModel",
    # Deterministic Gaussin Model
    "DeterministicGaussianModel",
    # SGLD Model.
    "SGLDModel",
)
