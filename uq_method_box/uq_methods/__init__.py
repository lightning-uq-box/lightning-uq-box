"""UQ-Methods as Lightning Modules."""

from .base import BaseModel
from .bnn_vi import BNN_VI, BNN_VI_Batched
from .bnn_vi_elbo import BNN_VI_ELBO
from .bnn_vi_lv import BNN_LV_VI, BNN_LV_VI_Batched
from .cqr_model import CQR
from .deep_ensemble_model import DeepEnsembleModel
from .deep_evidential_regression import DERModel
from .deterministic_gaussian import DeterministicGaussianModel
from .laplace_model import LaplaceModel
from .loss_functions import NLL, DERLoss, QuantileLoss
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
    # Deep Evidential Regression Model
    "DERModel",
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
