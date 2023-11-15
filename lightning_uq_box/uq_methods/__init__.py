"""UQ-Methods as Lightning Modules."""

from .base import BaseModule, DeterministicClassification, DeterministicModel
from .bnn_vi import BNN_VI_BatchedRegression, BNN_VIBase, BNN_VIREgression
from .bnn_vi_elbo import (
    BNN_VI_ELBO_Base,
    BNN_VI_ELBOClassification,
    BNN_VI_ELBORegression,
)
from .bnn_vi_lv import BNN_LV_VI, BNN_LV_VI_Batched
from .cards import CARDBase, NoiseScheduler
from .cqr_model import CQR
from .deep_ensemble import DeepEnsembleModel
from .deep_evidential_regression import DER
from .deep_kernel_learning import (
    DKLBase,
    DKLClassification,
    DKLGPLayer,
    DKLRegression,
    compute_initial_values,
)
from .deterministic_uncertainty_estimation import DUEClassification, DUERegression
from .laplace_model import LaplaceModel
from .loss_functions import NLL, DERLoss, HuberQLoss, QuantileLoss
from .mc_dropout import MCDropoutBase, MCDropoutClassification, MCDropoutRegression
from .mean_variance_estimation import MVEBase, MVERegression
from .quantile_regression import QuantileRegression, QuantileRegressionBase
from .sgld import SGLDModel
from .spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralNormConv,
    SpectralNormFC,
    spectral_normalize_model_layers,
)
from .swag import SWAGBase, SWAGClassification, SWAGRegression

__all__ = (
    # Base Module
    "BaseModule",
    # base model
    "DeterministicModel",
    "DeterministicClassification",
    # CARD Model
    "CARDBase",
    "NoiseScheduler",
    # conformalized Quantile Regression
    "CQR",
    # MC-Dropout
    "MCDropoutBase",
    "MCDropoutRegression",
    "MCDropoutClassification",
    # Laplace Approximation
    "LaplaceModel",
    # Quantile Regression
    "QuantileRegressionBase",
    "QuantileRegression",
    "QuantilePxRegression",
    # Deep Ensemble Wrapper
    "DeepEnsembleModel",
    # Mean Variance Estimation Network
    "MVEBase",
    "MVERegression",
    # Deep Uncertainty Estimation Model
    "DUERegression",
    "DUEClassification",
    # Deep Kernel Learning Model
    "DKLBase",
    "DKLRegression",
    "DKLClassification",
    # Approximate GP model for DKL
    "DKLGPLayer",
    "compute_initial_values",
    # SWAG Model
    "SWAGBase",
    "SWAGRegression",
    "SWAGClassification",
    # SGLD Model.
    "SGLDModel",
    # Deep Evidential Regression Model
    "DER",
    # Spectral Normalization Layers
    "SpectralBatchNorm1d",
    "SpectralBatchNorm2d",
    "SpectralNormConv",
    "SpectralNormFC",
    "spectral_normalize_model_layers",
    # BNN with ELBO
    "BNN_VI_ELBO_Base",
    "BNN_VI_ELBORegression",
    "BNN_VI_ELBOClassification",
    # Bayesian Neural Network trained with Variational Inference
    "BNN_VIBase",
    "BNN_VIREgression",
    "BNN_VI_BatchedRegression",
    # BNN with Latent Variables
    "BNN_LV_VI",
    "BNN_LV_VI_Batched",
    # Loss Functions
    "NLL",
    "QuantileLoss",
    "DERLoss",
    "HuberQLoss",
)
