"""UQ-Methods as Lightning Modules."""

from .base_model import BaseModel
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .laplace_model import LaplaceModel
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel

__all__ = (
    # base model
    "BaseModel",
    # Conformalized Quantile Regression
    "CQR",
    # MC-Dropout
    "MCDropoutModel",
    # Laplace Approximation
    "LaplaceModel",
    # Quantile Regression
    "QuantileRegressionModel",
    # Deep Ensemble Wrapper
    "DeepEnsembleModel",
)
