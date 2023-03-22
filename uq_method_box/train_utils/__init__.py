"""Train utilities for UQ-Regression-Box."""

from .loss_functions import MSE, NLL, QuantileLoss
from .train_scripts import basic_train_loop

__all__ = (
    # loss functions
    "NLL",
    "QuantileLoss",
    # train scripts
    "basic_train_loop",
    "MSE",
)
