"""Deep Uncertainty Estimation."""

from typing import Any, Dict, Union

import torch.nn as nn

from .base import BaseModel


class DUEModel(BaseModel):
    """Deep Uncertainty Estimation Model.

    If you use this model in your research, please cite the following paper:x

    * https://arxiv.org/abs/2102.11409
    """

    def __init__(
        self,
        model_class: Union[type[nn.Module], str],
        model_args: Dict[str, Any],
        lr: float,
        loss_fn: str,
        save_dir: str,
    ) -> None:
        """Initialize a new instance of Deep Uncertainty Estimation Model.

        Args:
            model_class:
            model_args:
            lr:
            loss_fn:
            save_dir:
        """
        super().__init__(model_class, model_args, lr, loss_fn, save_dir)
