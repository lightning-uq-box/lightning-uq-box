"""Implement a deep ensemble model, trained sequentially."""
# TODO should also support being able to load multiple checkpoints
# that can form an ensemble during inference

from typing import Any, Dict, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
)


class DeepEnsembleModel(LightningModule):
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
        self.config = config

        self.ensemble_members = ensemble_members

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            batch:

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        return torch.stack([member(args[0]) for member in self.ensemble_members], -1)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """Deep Ensemble only used for inference from trained models."""
        pass

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        preds = self.forward(batch)  # [batch_size, num_outputs, num_ensemble_members]

        mean_samples = preds[:, 0, :].detach().numpy()

        # assume nll prediction with sigma
        if preds.shape[1] == 2:
            sigma_samples = preds[:, 1, :].detach().numpy()
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
            }
        # assume mse prediction
        else:
            mean = mean_samples.mean(-1)
            std = mean_samples.std(-1)

            return {"mean": mean, "pred_uct": std, "epistemic_uct": std}
