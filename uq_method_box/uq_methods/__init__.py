"""UQ-Methods as Lightning Modules."""

from .base import BaseModel
from .bayes_by_backprop import BayesByBackpropModel
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .deep_evidential_regression import DERModel
from .deep_kernel_learning import DeepKernelLearningModel, DKLGPLayer, initial_values
from .deterministic_gaussian import DeterministicGaussianModel
from .deterministic_uncertainty_estimation import DUEModel
from .laplace_model import LaplaceModel
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel
from .spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralNormConv,
    SpectralNormFC,
    spectral_normalize_model_layers,
)

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
    # Deep Uncertainty Estimation Model
    "DUEModel",
    # Deep Kernel Learning Model
    "DeepKernelLearningModel",
    # Approximate GP model for DKL
    "DKLGPLayer",
    "initial_values",
    # Bayes by Backprop
    "BayesByBackpropModel",
    # Deep Evidential Regression Model
    "DERModel",
    # Spectral Normalization Layers
    "SpectralBatchNorm1d",
    "SpectralBatchNorm2d",
    "SpectralNormConv",
    "SpectralNormFC",
    "spectral_normalize_model_layers",
)
