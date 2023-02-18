"""UQ-Methods as Lightning Modules."""

from .base_model import BaseModel
from .cqr_model import CQR
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
)
