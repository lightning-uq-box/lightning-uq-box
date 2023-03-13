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
        """Define the Pyro model here."""
        return super().model(batch)

    def guide(self, batch):
        """Define the Pyro Guide here."""
        return super().guide(batch)

    def forward(self, x):
        """Put the part where the input is forwarded through the network here to produce an output."""  # noqa: E501
        return super().forward(x)
