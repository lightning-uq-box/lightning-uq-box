"""Train utilities for UQ-Regression-Box."""

from .loss_functions import NLL, QuantileLoss, TheirNLL
from .train_scripts import basic_train_loop

__all__ = (
    # loss functions
    "NLL",
    "TheirNLL",
    "QuantileLoss",
    # train scripts
    "basic_train_loop",
)
