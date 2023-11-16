"""Deep Evidential Regression."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .base import DeterministicModel
from .loss_functions import DERLoss
from .utils import _get_num_outputs, default_regression_metrics


class DERLayer(nn.Module):
    """Deep Evidential Regression Layer.


    Taken from `here <https://github.com/pasteurlabs/unreasonable_effective_der/blob/4631afcde895bdc7d0927b2682224f9a8a181b2c/models.py#L22>`_.
    """

    def __init__(self):
        """Initialize a new Deep Evidential Regression Layer."""
        super().__init__()
        self.in_features = 4
        self.out_features = 4

    def forward(self, x):
        """Compute the DER parameters.

        Args:
            x: feature output from network [batch_size x 4]

        Returns:
            DER outputs of shape [batch_size x 4]
        """
        assert x.dim() == 2, "Input X should be 2D."
        assert x.shape[-1] == 4, "DER method expects 4 inputs per sample."

        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class DER(DeterministicModel):
    """Deep Evidential Regression Model.

    Following the suggested implementation of `Unreasonable Effectiveness of Deep Evidential Regression <https://github.com/pasteurlabs/unreasonable_effective_der/blob/4631afcde895bdc7d0927b2682224f9a8a181b2c/models.py#L22`_.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2205.10060
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: type[Optimizer],
        lr_scheduler: type[LRScheduler] = None,
        coeff: float = 0.01,
    ) -> None:
        """Initialize a new Base Model.

        Args:
            model: pytorch model
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            coeff: coefficient for the DER loss
             from the predictive distribution
        """
        super().__init__(model, optimizer, None, lr_scheduler)

        self.save_hyperparameters(ignore=["model"])

        # check that output is 4 dimensional
        assert _get_num_outputs(model) == 4, "DER model expects 4 outputs."

        # add DER Layer
        self.model = nn.Sequential(self.model, DERLayer())

        # set DER Loss
        # TODO need to give control over the coeff through config or argument
        self.loss_fn = DERLoss(coeff)

        self.hparams["quantiles"] = quantiles

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Prediction Step Deep Evidential Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        Returns:
            dictionary with predictions and uncertainty measures
        """
        with torch.no_grad():
            pred = self.model(X)  # [batch_size x 4]
            pred_np = pred.cpu().numpy()

        gamma, nu, alpha, beta = (
            pred[:, 0:1],
            pred_np[:, 1],
            pred_np[:, 2],
            pred_np[:, 3],
        )

        epistemic_uct = self.compute_epistemic_uct(nu)
        aleatoric_uct = self.compute_aleatoric_uct(beta, alpha, nu)
        pred_uct = epistemic_uct + aleatoric_uct

        quantiles = compute_quantiles_from_std(
            gamma.squeeze(-1).cpu().numpy(), pred_uct, self.hparams.quantiles
        )

        return {
            "pred": gamma,
            "pred_uct": pred_uct,
            "aleatoric_uct": aleatoric_uct,
            "epistemic_uct": epistemic_uct,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, 1],
            "out": pred,
        }

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from 4D model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x 4]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        return out[:, 0:1]

    def compute_aleatoric_uct(
        self, beta: np.ndarray, alpha: np.ndarray, nu: np.ndarray
    ) -> Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10:

        Args:
            beta: beta output DER model
            alpha: alpha output DER model
            nu: nu output DER model

        Returns:
            Aleatoric Uncertainty
        """
        # Equation 10 from the above paper
        return np.sqrt(np.divide(beta * (1 + nu), alpha * nu))

    def compute_epistemic_uct(self, nu: np.ndarray) -> np.ndarray:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10:

        Args:
            nu: nu output DER model
        Returns:
            Epistemic Uncertainty
        """
        return np.reciprocal(np.sqrt(nu))
