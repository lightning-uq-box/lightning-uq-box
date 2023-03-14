"""Implement a deep ensemble model, trained sequentially."""
# TODO should also support being able to load multiple checkpoints
# that can form an ensemble during inference

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from uq_method_box.uq_methods import EnsembleModel


class DeepEnsembleModel(EnsembleModel):
    """Deep Ensemble Wrapper.

    Should accept a list of instantiated models and run
    inference for them.
    """

    def __init__(
        self, config: Dict[str, Any], ensemble_members: List[nn.Module]
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Checkpoint files should be pytorch model checkpoints and
        not lightning checkpoints.
        """
        super().__init__(config, ensemble_members)

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        return torch.stack([member(X) for member in self.ensemble_members], -1)

    def generate_ensemble_predictions(self, X: Tensor) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            the ensemble predictions
        """
        return self.forward(X)  # [batch_size, num_outputs, num_ensemble_members]
