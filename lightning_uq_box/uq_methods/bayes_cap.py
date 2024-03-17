"""BayesCap model for uncertainty quantification."""

# Adapted from https://github.com/ExplainableML/BayesCap

from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import BaseModule, DeterministicRegression
from .utils import default_regression_metrics


class BayesCap(DeterministicRegression):
    """BayesCap model for uncertainty quantification.

    If you use this model, please cite the following paper:

    * https://arxiv.org/abs/2207.06873
    """

    def __init__(
        self,
        model: nn.Module,
        bayes_cap_model: nn.Module,
        loss_fn: nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initializes the BayesCap model.

        Args:
            model: the pretrained frozen model
            bayes_cap_model: the BayesCap model to be trained to
                quantify the model's uncertainty
            loss_fn: the loss function to use for training
            optimizer: the optimizer to use for training
            lr_scheduler: the learning rate scheduler to use for training
        """
        super().__init__(bayes_cap_model, loss_fn, optimizer, lr_scheduler)

        self.model = model
        self.bayes_cap_model = bayes_cap_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    @torch.no_grad()
    def forward_base_model(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass of the base model.

        This can be overwritten to adapt to various tasks like
        inpainting, superrresolution, etc. where the forward pass
        may require additional arguments.

        Args:
            batch: the output of your DataLoader

        Returns:
            the output of the base model
        """
        return self.model(batch[self.input_key])

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        base_model_output = self.forward_base_model(batch)
        bayes_cap_input = base_model_output.clone()

        mu, alpha, beta = self.bayes_cap_model(bayes_cap_input)

        loss = self.loss_fn(mu, alpha, beta, bayes_cap_input, batch[self.target_key])

        self.log("train_loss", loss, batch_size=batch[self.input_key].size(0))
        if batch[self.input_key].size(0) == 1:
            self.train_metrics(mu, batch[self.target_key])

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        base_model_output = self.forward_base_model(batch)
        bayes_cap_input = base_model_output.clone()

        mu, alpha, beta = self.bayes_cap_model(bayes_cap_input)

        loss = self.loss_fn(mu, alpha, beta, bayes_cap_input, batch[self.target_key])

        self.log("val_loss", loss, batch_size=batch[self.input_key].size(0))
        if batch[self.input_key].size(0) == 1:
            self.val_metrics(mu, batch[self.target_key])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1)

        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                pred_dict["pred"].squeeze(-1), batch[self.target_key].squeeze(-1)
            )

        # turn mean to np array
        pred_dict["pred"] = pred_dict["pred"].detach().cpu().squeeze(-1)

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        return pred_dict

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        # TODO need to come up with good solution for segmentation and image data
        # maybe h5 that stores the predict output for each separate sample with idx
        # by default turned off because probably storage intensive
        pass

    def predict_step(
        self, X: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Predict the output of the model.

        Args:
            X: the input data
            batch_idx: the index of this batch
            dataloder_idx: the index of the dataloader

        Returns:
            the model's prediction
        """
        with torch.no_grad():
            base_model_output = self.model(X)
            mu, alpha, beta = self.bayes_cap_model(base_model_output)

        a_map = (1 / (alpha + 1e-5)).to("cpu").data
        b_map = beta.to("cpu").data

        pred_uct = (a_map**2) * (
            torch.exp(torch.lgamma(3 / (b_map + 1e-2)))
            / torch.exp(torch.lgamma(1 / (b_map + 1e-2)))
        )

        return {"pred": mu, "alpha": alpha, "beta": beta, "pred_uct": pred_uct}

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.bayes_cap_model())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}
