"""Implement a deep ensemble model, trained sequentially."""
# TODO should also support being able to load multiple checkpoints
# that can form an ensemble during inference

from typing import Any, Dict, List, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from uq_method_box.uq_methods import EnsembleModel


class DeepEnsembleModel(EnsembleModel):
    """Deep Ensemble Wrapper.

    Should accept a list of instantiated models and run
    inference for them.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ensemble_members: List[Dict[str, Union[type[LightningModule], str]]],
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            config: configuration dictionary
            ensemble_members: List of dicts where each element specifies the
                LightningModule class and a path to a checkpoint

        Checkpoint files should be pytorch model checkpoints and
        not lightning checkpoints.
        """
        super().__init__(config, ensemble_members)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass over ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        out: List[torch.Tensor] = []
        for model_config in self.ensemble_members:
            loaded_model = model_config["model_class"].load_from_checkpoint(
                model_config["ckpt_path"]
            )
            out.append(loaded_model(X))
        return torch.stack(out, dim=-1)

    def generate_ensemble_predictions(self, X: Tensor) -> Tensor:
        """Generate DeepEnsemble Predictions.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            the ensemble predictions
        """
        return self.forward(X)  # [batch_size, num_outputs, num_ensemble_members]
