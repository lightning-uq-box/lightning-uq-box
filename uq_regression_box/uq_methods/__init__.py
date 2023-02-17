"""UQ-Methods as Lightning Modules."""

from .base_model import BaseModel
from .cqr_model import CQR

__all__ = (
    # base model
    "BaseModel",
    # Conformalized Quantile Regression
    "CQR",
)
