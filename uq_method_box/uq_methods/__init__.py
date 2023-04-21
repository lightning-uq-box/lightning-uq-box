"""UQ-Methods as Lightning Modules."""

from .base import BaseModel
from .bayes_by_backprop import BayesByBackpropModel
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .deep_evidential_regression import DERModel
from .deterministic_gaussian import DeterministicGaussianModel
from .laplace_model import LaplaceModel
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel
from .sgld import SGLDModel
from .swag import SWAGModel

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
    # Deterministic Gaussian Model
    "DeterministicGaussianModel",
    # SWAG Model
    "SWAGModel",
    # SGLD Model.
    "SGLDModel",
    # Bayes by Backprop
    "BayesByBackpropModel",
    # Deep Evidential Regression Model
    "DERModel",
)
