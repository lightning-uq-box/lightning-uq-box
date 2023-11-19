"""UQ-Methods as Lightning Modules."""

from .base import (
    BaseModule,
    DeterministicClassification,
    DeterministicModel,
    DeterministicRegression,
    PosthocBase,
)
from .bnn_vi import BNN_VI_Base, BNN_VI_BatchedRegression, BNN_VI_Regression
from .bnn_vi_elbo import (
    BNN_VI_ELBO_Base,
    BNN_VI_ELBO_Classification,
    BNN_VI_ELBO_Regression,
)
from .bnn_vi_lv import BNN_LV_VI, BNN_LV_VI_Batched
from .cards import CARDBase, CARDClassification, CARDRegression, NoiseScheduler
from .cqr_model import ConformalQR
from .deep_ensemble import (
    DeepEnsemble,
    DeepEnsembleClassification,
    DeepEnsembleRegression,
)
from .deep_evidential_regression import DER
from .deep_kernel_learning import (
    DKLBase,
    DKLClassification,
    DKLGPLayer,
    DKLRegression,
    compute_initial_values,
)
from .deterministic_uncertainty_estimation import DUEClassification, DUERegression
from .laplace_model import LaplaceBase, LaplaceClassification, LaplaceRegression
from .loss_functions import NLL, DERLoss, HuberQLoss, QuantileLoss
from .mc_dropout import MCDropoutBase, MCDropoutClassification, MCDropoutRegression
from .mean_variance_estimation import MVEBase, MVERegression
from .quantile_regression import QuantileRegression, QuantileRegressionBase
from .sgld import SGLDBase, SGLDClassification, SGLDRegression
from .spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralNormConv,
    SpectralNormFC,
    spectral_normalize_model_layers,
)
from .swag import SWAGBase, SWAGClassification, SWAGRegression
from .temp_scaling import TempScaling

__all__ = (
    # Base Module
    "BaseModule",
    "PosthocBase",
    # base model
    "DeterministicModel",
    "DeterministicClassification",
    "DeterministicRegression",
    # CARD Model
    "CARDBase",
    "CARDRegression",
    "CARDClassification",
    "NoiseScheduler",
    # conformalized Quantile Regression
    "ConformalQR",
    # MC-Dropout
    "MCDropoutBase",
    "MCDropoutRegression",
    "MCDropoutClassification",
    # Laplace Approximation
    "LaplaceBase",
    "LaplaceRegression",
    "LaplaceClassification",
    # Quantile Regression
    "QuantileRegressionBase",
    "QuantileRegression",
    "QuantilePxRegression",
    # Deep Ensemble Wrapper
    "DeepEnsemble",
    "DeepEnsembleRegression",
    "DeepEnsembleClassification",
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
    "SGLDBase",
    "SGLDRegression",
    "SGLDClassification",
    # Temperature Scaling
    "TempScaling",
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
    "BNN_VI_ELBO_Regression",
    "BNN_VI_ELBO_Classification",
    # Bayesian Neural Network trained with Variational Inference
    "BNN_VI_Base",
    "BNN_VI_Regression",
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
