"""UQ-Methods as Lightning Modules."""

from .base import BaseModel
from .bnn_vi import BNN_VI, BNN_VI_Batched
from .bnn_vi_elbo import BNN_VI_ELBO
from .bnn_vi_lv import BNN_LV_VI, BNN_LV_VI_Batched
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .deep_evidential_regression import DERModel
from .deep_kernel_learning import DeepKernelLearningModel, DKLGPLayer, initial_values
from .deterministic_gaussian import DeterministicGaussianModel
from .deterministic_uncertainty_estimation import DUEModel
from .laplace_model import LaplaceModel
from .loss_functions import NLL, DERLoss, QuantileLoss
from .mc_dropout_model import MCDropoutModel
from .quantile_regression_model import QuantileRegressionModel
from .sgld import SGLDModel
from .spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralNormConv,
    SpectralNormFC,
    spectral_normalize_model_layers,
)
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
    # Deep Uncertainty Estimation Model
    "DUEModel",
    # Deep Kernel Learning Model
    "DeepKernelLearningModel",
    # Approximate GP model for DKL
    "DKLGPLayer",
    "initial_values",
    # SWAG Model
    "SWAGModel",
    # SGLD Model.
    "SGLDModel",
    # Deep Evidential Regression Model
    "DERModel",
    # Spectral Normalization Layers
    "SpectralBatchNorm1d",
    "SpectralBatchNorm2d",
    "SpectralNormConv",
    "SpectralNormFC",
    "spectral_normalize_model_layers",
    # BNN with ELBO
    "BNN_VI_ELBO",
    # Bayesian Neural Network trained with Variational Inference
    "BNN_VI",
    "BNN_VI_Batched",
    # BNN with Latent Variables
    "BNN_LV_VI",
    "BNN_LV_VI_Batched",
    # Loss Functions
    "NLL",
    "QuantileLoss",
    "DERLoss",
)
