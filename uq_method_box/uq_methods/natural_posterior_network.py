"""Natural Posterior Network."""

import os
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch import Callback
from natpn.model.lightning_module_flow import NaturalPosteriorNetworkFlowLightningModule
from natpn.nn import BayesianLoss, NaturalPosteriorNetworkModel
from natpn.nn.flow import NormalizingFlow
from natpn.nn.output import NormalOutput
from pytorch_lightning.callbacks import EarlyStopping
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
        loss = self.criterion.forward(posterior, y_true.squeeze(-1))

        self.log("train_loss", loss, prog_bar=True)  # logging to Logger
        self.train_metrics(mean_pred, y_true.squeeze(-1))

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Batch, _batch_idx: int = 0) -> Tensor:
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
        loss = self.criterion(posterior, y_true.squeeze(-1))

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
        aleatoric = -posterior.maximum_a_posteriori().uncertainty().cpu().numpy()
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
        return {"optimizer": optimizer}


class NaturalPosteriorNetworkFlow(LightningModule):
    """
    Lightning module for optimizing the normalizing flow of NatPN.
    """

    def __init__(
        self,
        model: NaturalPosteriorNetworkModel,
        learning_rate: float = 1e-3,
        learning_rate_decay: bool = False,
        early_stopping: bool = True,
    ) -> None:
        """Initialize a new instance of NPN Flow

        Args:
            model: The model whose flow to optimize.
            learning_rate: The learning rate to use for the Adam optimizer.
            learning_rate_decay: Whether to use a learning rate decay. If set to ``True``, the
                learning rate schedule is implemented using
                :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.
            early_stopping: Whether to use early stopping for training.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.early_stopping = early_stopping

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.model.flow.parameters(), lr=self.learning_rate)
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self.learning_rate_decay:
            config["lr_scheduler"] = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.25,
                    patience=self.trainer.max_epochs // 20,
                    threshold=1e-3,
                    min_lr=1e-7,
                ),
                "monitor": "val/log_prob",
            }
        return config

    # def configure_callbacks(self) -> List[Callback]:
    #     if not self.early_stopping:
    #         return []
    #     return [
    #         EarlyStopping(
    #             "val/log_prob",
    #             min_delta=1e-2,
    #             mode="max",
    #             patience=self.trainer.max_epochs // 10,
    #         ),
    #     ]

    def training_step(self, batch: Batch, _batch_idx: int) -> torch.Tensor:
        X, _ = batch
        log_prob = self.model.log_prob(X, track_encoder_gradients=False).mean()
        self.log("train/log_prob", log_prob, prog_bar=True)
        return -log_prob

    def validation_step(self, batch: Batch, _batch_idx: int = 0) -> None:
        X, _ = batch
        log_prob = self.model.log_prob(X).mean()
        self.log("val/log_prob", log_prob, prog_bar=True)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """NPN Predict step after finetuning Flow Network.

        Args:
            X: input tensor

        Returns:
            prediction dictionary
        """
        with torch.no_grad():
            posterior, log_prob = self.model.forward(X)

        mean = posterior.maximum_a_posteriori().mean().cpu().numpy()

        # log_prob is epistemic uncertainty
        # https://github.com/borchero/natural-posterior-network/blob/main/natpn/model/lightning_module_ood.py#L45
        epistemic = torch.clone(log_prob).cpu().numpy()

        # posterior map uncertainty is aleatoric uncertainty
        # https://github.com/borchero/natural-posterior-network/blob/main/natpn/model/lightning_module_ood.py#L35
        aleatoric = -posterior.maximum_a_posteriori().uncertainty().cpu().numpy()
        return {
            "mean": mean,
            "pred_uct": epistemic + aleatoric,
            "epistemic_uct": epistemic,
            "aleatoric_uct": aleatoric,
        }
