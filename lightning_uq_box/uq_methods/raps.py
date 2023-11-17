"""Regularized Adaptive Prediction Sets (RAPS)."""

"""Adapted from https://github.com/aangelopoulos/conformal_classification"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .base import PosthocBase


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    def __init__(self, model: Union[LightningModule, nn.Module]) -> None:
        """Initialize RAPS.

        Args:
            model: A PyTorch Lightning or PyTorch module
        """
        super().__init__(model)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step after running posthoc fitting methodology."""
        raise NotImplementedError

    def adjust_model_output(self, model_output: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_output: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model output tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError
