"""UQ-Methods as Lightning Modules."""

from .base import BaseModel, EnsembleModel
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .gaussian_nll import DeterministicGaussianModel
from .laplace_model import LaplaceModel
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel

__all__ = (
    # base model
    "BaseModel",
    "EnsembleModel",
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
)
