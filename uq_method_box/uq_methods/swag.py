"""Stochastic Weight Averaging - Gaussian."""

from typing import Any, Dict

from .base import BaseModel
from .tracker import SWAGTracker


class SWAGModel(BaseModel):
    """Stochastic Weight Averaging - Gaussian (SWAG)."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module = None,
        criterion: nn.Module = ...,
    ) -> None:
        super().__init__(config, model, criterion)
