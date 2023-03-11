"""Bayesian Neural Network with Latent Variables."""

from typing import Any, Dict

from .base import PyroBNNBase


class BNN_LV(PyroBNNBase):
    """Bayesian Neural Network with Latent Variables.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/1710.07283
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize a new instance of BNN+LV.

        Args:
            config:
        """
        super().__init__(config)

    def model(self, batch):
        return super().model(batch)

    def guide(self, batch):
        return super().guide(batch)

    def forward(self, x):
        return super().forward(x)
