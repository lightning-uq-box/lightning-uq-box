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

    def forward(self, batch: Any, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            batch:

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        X = batch[0]
        return torch.stack([member(X) for member in self.ensemble_members], -1)

    def generate_ensemble_predictions(self, batch: Any) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            batch:

        Returns:
            the ensemble predictions
        """
        return self.forward(batch)  # [batch_size, num_outputs, num_ensemble_members]
