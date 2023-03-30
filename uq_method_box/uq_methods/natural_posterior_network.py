"""Natural Posterior Network."""

import os
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from natpn.nn.flow import NormalizingFlow
from natpn.nn.output import NormalOutput
from torch import Tensor, optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from .utils import save_predictions_to_csv

"""
A reference to a flow type that can be used with :class:`NaturalPosteriorNetwork`:
- `radial`: A :class:`~natpn.nn.flow.RadialFlow`.
- `maf`: A :class:`~natpn.nn.flow.MaskedAutoregressiveFlow`.
"""

CertaintyBudget = Literal["constant", "exp-half", "exp", "normal"]
"""
The certainty budget to distribute in the latent space of dimension ``H``:
- ``constant``: A certainty budget of 1, independent of the latent space's dimension.
- ``exp-half``: A certainty budget of ``exp(0.5 * H)``.
- ``exp``: A certainty budget of ``exp(H)``.
- ``normal``: A certainty budget that causes a multivariate normal to yield the same
  probability at the origin at any dimension: ``exp(0.5 * log(4 * pi) * H)``.
"""

Batch = Tuple[torch.Tensor, torch.Tensor]


class NaturalPosteriorNetwork(LightningModule):
    """Natural Posterior Network.

    Implementation based on and building on
    https://github.com/borchero/natural-posterior-network.

    If you use this method in your research, please cite the following paper:

    * https://arxiv.org/abs/2105.04471
    """

    def __init__(
        self,
        latent_dim: int,
        encoder: nn.Module,  # this encoder like other feature extractors like DKL?
        flow: type[NormalizingFlow],
        flow_num_layers: int,
        certainty_budget: CertaintyBudget,
        entropy_weight: float,
        lr: float,
        save_dir: str,
    ) -> None:
        """Initialize a new instance of Natural Posterior Network.

        Args:
            latent_dim:
            encoder:
            flow: Normalizing Flow class
            flow_num_layers: number of layers in normalizing flow
            certainty_budget:
            entropy_weight: entropy weight for loss function
            lr: learning rate for Adam optimizer
            save_dir: directory where to store predictions
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.model = NaturalPosteriorNetworkModel(
            latent_dim=latent_dim,
            encoder=encoder,
            flow=flow(latent_dim, flow_num_layers),
            output=NormalOutput(latent_dim),
            certainty_budget=certainty_budget,
        )

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="test_",
        )

        self.criterion = BayesianLoss(entropy_weight)

    def training_step(self, batch: Batch, _batch_idx: int) -> Tensor:
        """Compute training step for NPN.

        Args:
            batch: batch from dataloader

        Returns:
            computed loss
        """
        X, y_true = batch
        posterior, log_prob = self.model.forward(X)
        mean_pred = posterior.maximum_a_posteriori().mean()
        loss = self.criterion(posterior, y_true)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(mean_pred, y_true.squeeze(-1))

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Batch, _batch_idx: int) -> Tensor:
        """Compute validation step.

        Args:
            batch: batch from dataloader

        Returns:
            validation loss # TODO is this needed?
        """
        X, y_true = batch
        # posterior is a natpn posterior object
        posterior, log_prob = self.model.forward(X)
        mean_pred = posterior.maximum_a_posteriori().mean()
        loss = self.criterion(posterior, y_true)

        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(mean_pred, y_true.squeeze(-1))
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y_true = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y_true.detach().squeeze(-1).cpu().numpy()
        return out_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
        """
        with torch.no_grad():
            posterior, log_prob = self.model.forward(X)

        mean = posterior.maximum_a_posteriori().mean().cpu().numpy()

        # log_prob is epistemic uncertainty
        # https://github.com/borchero/natural-posterior-network/blob/main/natpn/model/lightning_module_ood.py#L45
        epistemic = torch.clone(log_prob).cpu().numpy()

        # posterior map uncertainty is aleatoric uncertainty
        # https://github.com/borchero/natural-posterior-network/blob/main/natpn/model/lightning_module_ood.py#L35
        aleatoric = posterior.maximum_a_posteriori().uncertainty().cpu().numpy()
        return {
            "mean": mean,
            "pred_uct": epistemic + aleatoric,
            "epistemic_uct": epistemic,
            "aleatoric_uct": aleatoric,
        }

    def on_test_batch_end(
        self,
        outputs: Dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        save_predictions_to_csv(
            outputs, os.path.join(self.hparams.save_dir, "predictions.csv")
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        # config: Dict[str, Any] = {"optimizer": optimizer}
        # if self.learning_rate_decay:
        #     config["lr_scheduler"] = {
        #         "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer,
        #             factor=0.25,
        #             patience=self.trainer.max_epochs // 20,
        #             threshold=1e-3,
        #             min_lr=1e-7,
        #         ),
        #         "monitor": "val/loss",
        #     }
        return {"optimizer": optimizer}
