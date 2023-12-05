"""Temperature Scaling.

Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py. # noqa: E501
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import LBFGS

from .base import PosthocBase


class TempScaling(PosthocBase):
    """Temperature Scaling.

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/1706.04599
    """

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        optim_lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """Initialize Temperature Scaling method.

        Args:
            model: model to be calibrated with Temperature S
            optim_lr: learning rate for optimizer
            max_iter: maximum number of iterations to run optimizer
        """
        super().__init__(model)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optim_lr = optim_lr
        self.max_iter = max_iter
        self.criterion = nn.CrossEntropyLoss()

    @torch.enable_grad()
    def adjust_model_logits(self, model_logits: Tensor) -> Tensor:
        """Adjust model logits by applying temperature scaling.

        Args:
            model_logits: model output logits of shape [batch_size x num_outputs]

        Returns:
            adjusted model logits of shape [batch_size x num_outputs]
        """
        temperature = self.temperature.unsqueeze(1).expand(
            model_logits.size(0), model_logits.size(1)
        )
        return model_logits / temperature

    def on_validation_epoch_end(self) -> None:
        """Perform CQR computation to obtain q_hat for predictions.

        Args:
            outputs: list of dictionaries containing model outputs and labels

        """
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        # optimizer temperature w.r.t. NLL
        optimizer = LBFGS([self.temperature], lr=self.optim_lr, max_iter=self.max_iter)

        # also lightning automatically disables gradient computation during this stage
        # but need it for temp scaling optimization so set inference mode to false with
        # context manager
        with torch.inference_mode(False):
            all_logits = all_logits.clone().requires_grad_(True)

            def eval():
                optimizer.zero_grad()
                loss = self.criterion(self.adjust_model_logits(all_logits), all_labels)
                loss.backward()
                return loss

            optimizer.step(eval)

        self.post_hoc_fitted = True

    def predict_step(self, X: Tensor) -> Dict[str, Tensor]:
        """Prediction step with applied temperature scaling.

        Args:
            X: input tensor of shape [batch_size x num_features]
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, please call trainer.fit(model, datamodule) first."  # noqa: E501
            )
        with torch.no_grad():
            temp_scaled_outputs = self.forward(X)
        entropy = -torch.sum(
            F.softmax(temp_scaled_outputs, dim=1)
            * F.log_softmax(temp_scaled_outputs, dim=1),
            dim=1,
        )

        return {"pred": temp_scaled_outputs, "pred_uct": entropy}

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step after running posthoc fitting methodology.

        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        preds = self.predict_step(batch[self.input_key])
        return preds
