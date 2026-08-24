"""BayesCap model for uncertainty quantification."""

# Adapted from https://github.com/ExplainableML/BayesCap

from typing import Any

import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn

from .base import DeterministicRegression
from .loss_functions import TempCombLoss
from .utils import _get_num_outputs


class BayesCapLayer(nn.Module):
    """BayesCap Layer.

    Splits a raw 3-channel model output into the `(mu, one_over_alpha, beta)`
    parameters of the generalized Gaussian predictive distribution.
    """

    def __init__(self):
        """Initialize a new BayesCap Layer."""
        super().__init__()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the BayesCap parameters.

        Args:
            x: feature output from network [batch_size x 3]

        Returns:
            mu, one_over_alpha, and beta, each of shape [batch_size x 1]
        """
        assert x.shape[1] == 3, "BayesCap method expects 3 input features per sample."

        mu = x[:, 0:1, ...]
        one_over_alpha = nn.functional.softplus(x[:, 1:2, ...])
        beta = nn.functional.softplus(x[:, 2:3, ...])
        return mu, one_over_alpha, beta


class BayesCap(DeterministicRegression):
    """BayesCap model for uncertainty quantification.

    If you use this model, please cite the following paper:

    * https://arxiv.org/abs/2207.06873
    """

    def __init__(
        self,
        model: nn.Module,
        bayes_cap_model: nn.Module,
        loss_fn: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initializes the BayesCap model.

        Args:
            model: the pretrained frozen model
            bayes_cap_model: the BayesCap model to be trained to
                quantify the model's uncertainty
            loss_fn: the loss function to use for training, defaults to
                :class:`~lightning_uq_box.uq_methods.loss_functions.TempCombLoss`
            optimizer: the optimizer to use for training
            lr_scheduler: the learning rate scheduler to use for training
        """
        assert _get_num_outputs(bayes_cap_model) == 3, (
            "BayesCap model expects 3 outputs."
        )
        loss_fn = loss_fn or TempCombLoss()

        super().__init__(
            bayes_cap_model, loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

        self.model = model
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.bayes_cap_model = bayes_cap_model
        self.bayes_cap_layer = BayesCapLayer()

    def train(self, mode: bool = True) -> "BayesCap":
        """Set the module in training mode, keeping the base model frozen.

        Args:
            mode: whether to set training mode (True) or evaluation mode (False)

        Returns:
            self
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            X: input data

        Returns:
            mu, one_over_alpha, beta, and the frozen base model's output
        """
        with torch.no_grad():
            base_model_output = self.model(X)

        mu, one_over_alpha, beta = self.bayes_cap_layer(
            self.bayes_cap_model(base_model_output)
        )
        return mu, one_over_alpha, beta, base_model_output

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        mu, one_over_alpha, beta, base_model_output = self.forward(
            batch[self.input_key]
        )
        loss = self.loss_fn(
            mu, one_over_alpha, beta, base_model_output, batch[self.target_key]
        )

        self.log("train_loss", loss, batch_size=batch[self.input_key].size(0))
        if batch[self.input_key].size(0) > 1:
            self.train_metrics(mu, batch[self.target_key])

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        mu, one_over_alpha, beta, base_model_output = self.forward(
            batch[self.input_key]
        )
        loss = self.loss_fn(
            mu, one_over_alpha, beta, base_model_output, batch[self.target_key]
        )

        self.log("val_loss", loss, batch_size=batch[self.input_key].size(0))
        if batch[self.input_key].size(0) > 1:
            self.val_metrics(mu, batch[self.target_key])

        return loss

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict the output of the model.

        Args:
            X: the input data
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            the model's prediction
        """
        with torch.no_grad():
            mu, one_over_alpha, beta, _ = self.forward(X)

        a_map = (1 / (one_over_alpha + 1e-5)).to("cpu").data
        b_map = beta.to("cpu").data

        pred_uct = (a_map**2) * (
            torch.exp(torch.lgamma(3 / (b_map + 1e-2)))
            / torch.exp(torch.lgamma(1 / (b_map + 1e-2)))
        )

        return {"pred": mu, "alpha": one_over_alpha, "beta": beta, "pred_uct": pred_uct}

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.bayes_cap_model.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
