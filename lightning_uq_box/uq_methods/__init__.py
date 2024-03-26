# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""UQ-Methods as Lightning Modules."""

from .base import (
    BaseModule,
    DeterministicClassification,
    DeterministicModel,
    DeterministicPixelRegression,
    DeterministicRegression,
    DeterministicSegmentation,
    PosthocBase,
)
from .bnn_lv_vi import (
    BNN_LV_VI_Base,
    BNN_LV_VI_Batched_Base,
    BNN_LV_VI_Batched_Regression,
    BNN_LV_VI_Regression,
)
from .bnn_vi import BNN_VI_Base, BNN_VI_BatchedRegression, BNN_VI_Regression
from .bnn_vi_elbo import (
    BNN_VI_ELBO_Base,
    BNN_VI_ELBO_Classification,
    BNN_VI_ELBO_Regression,
    BNN_VI_ELBO_Segmentation,
)
from .cards import CARDBase, CARDClassification, CARDRegression, NoiseScheduler
from .conformal_qr import ConformalQR
from .deep_ensemble import (
    DeepEnsemble,
    DeepEnsembleClassification,
    DeepEnsembleRegression,
    DeepEnsembleSegmentation,
)
from .deep_evidential_regression import DER, DERPxRegression
from .deep_kernel_learning import (
    DKLBase,
    DKLClassification,
    DKLGPLayer,
    DKLRegression,
    compute_initial_values,
)
from .deterministic_uncertainty_estimation import DUEClassification, DUERegression
from .hierarchical_prob_unet import HierarchicalProbUNet
from .img2img_conformal import Img2ImgConformal
from .inference_time_augmentation import TTABase, TTAClassification, TTARegression
from .laplace_model import LaplaceBase, LaplaceClassification, LaplaceRegression
from .loss_functions import NLL, DERLoss, PinballLoss
from .mc_dropout import (
    MCDropoutBase,
    MCDropoutClassification,
    MCDropoutPxRegression,
    MCDropoutRegression,
    MCDropoutSegmentation,
)
from .mean_variance_estimation import MVEBase, MVEPxRegression, MVERegression
from .prob_unet import ProbUNet
from .quantile_regression import (
    QuantilePxRegression,
    QuantileRegression,
    QuantileRegressionBase,
)
from .raps import RAPS
from .sgld import SGLDBase, SGLDClassification, SGLDRegression
from .sngp import SNGPBase, SNGPClassification, SNGPRegression
from .spectral_normalized_layers import (
    SpectralBatchNorm1d,
    SpectralBatchNorm2d,
    SpectralNormConv,
    SpectralNormFC,
    spectral_normalize_model_layers,
)
from .swag import SWAGBase, SWAGClassification, SWAGRegression, SWAGSegmentation
from .temp_scaling import TempScaling

__all__ = (
    # Base Module
    "BaseModule",
    "PosthocBase",
    # base model
    "DeterministicModel",
    "DeterministicClassification",
    "DeterministicRegression",
    "DeterministicSegmentation",
    "DeterministicPixelRegression",
    # CARDS
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
    "MCDropoutSegmentation",
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
    "DeepEnsembleSegmentation",
    # Mean Variance Estimation Network
    "MVEBase",
    "MVERegression",
    "MVEPxRegression",
    "MCDropoutPxRegression",
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
    "SWAGSegmentation",
    # SGLD Model.
    "SGLDBase",
    "SGLDRegression",
    "SGLDClassification",
    # SNGP Model
    "SNGPBase",
    "SNGPRegression",
    "SNGPClassification",
    # RAPS Model
    "RAPS",
    # Temperature Scaling
    "TempScaling",
    # Deep Evidential Regression Model
    "DER",
    "DERPxRegression",
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
    "BNN_VI_ELBO_Segmentation",
    # Bayesian Neural Network trained with Variational Inference
    "BNN_VI_Base",
    "BNN_VI_Regression",
    "BNN_VI_BatchedRegression",
    # BNN with Latent Variables
    "BNN_LV_VI_Base",
    "BNN_LV_VI_Regression",
    "BNN_LV_VI_Batched_Base",
    "BNN_LV_VI_Batched_Regression",
    # Image-to-Image Conformal Uncertainty Estimation
    "Img2ImgConformal",
    # Probabilistic Unet
    "ProbUNet",
    # Hierarchical Probabilistic Unet
    "HierarchicalProbUNet",
    # Loss Functions
    "NLL",
    "DERLoss",
    "PinballLoss",
    # Test time augmentation
    "TTABase",
    "TTARegression",
    "TTAClassification",
)
