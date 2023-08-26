"""Conditional Neural Process."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .base import BaseModel

# TODO: https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/dataset/dataset.py
# TODO: write appropriate Data Generator
# TODO:

# Exlanation Conditional Neural Process
# NPs model the mapping from datasets to parameters directly using Deep Sets Theory
# Encoder: Set Encoder takes in pairs of input and target samples, so all N points in the datasets
#   pass them through an encoder which yields a vector representation over which you take a sum
#   which distills a dataset representation (mapping from dataset to a vector)

# Decoder: function rho, where we also pass in the query location (location at which we want to make a prediction)
#   and together with the encoder representation we pass it through the decoder to get a mean and std
#   representation for each query location (mapping from a vector to mean/var), use decoder to query arbitrary test locations

# Training Procedure


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


class ConvCondNeuralProcess(BaseModel):
    """Convolutional Conditional Neural Process."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mean_layer: nn.Module,
        sigma_layer: nn.Module,
        points_per_unit: int,
        model: nn.Module,
        optimizer: type[Optimizer],
        loss_fn: nn.Module,
        lr_scheduler: type[LRScheduler] = None,
        save_dir: str = None,
    ) -> None:
        """Initialize a new instance of Conditional Neural Process.

        Args:
            encoder:
            decoder: denoted rho in the paper
            mean_layer: map decoder to mean output
            sigma_layer: map decoder to sigma output
            points_per_unit: number of points per unit interval on the input
                and used to discretize the function
        """
        super().__init__(model, optimizer, loss_fn, lr_scheduler, save_dir)

        self.encoder = encoder
        self.decoder = decoder
        self.mean_layer = mean_layer
        self.sigma_layer = sigma_layer

        self.points_per_unit = points_per_unit

    def forward(self, x_context: Tensor, y_context: Tensor, x_target: Tensor) -> Tensor:
        """Forward pass of the Model.

        Args:
            x_context: observation locations of shape [batch, data, features]
            y_context: observation values of shape [batch, data, features]
            x_target: locations of outputs of shape [batch, data, features]

        Returns:
            means and stds of shape [batch, 2]
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = (
            min(
                torch.min(x_context).cpu().numpy(),
                torch.min(x_target).cpu().numpy(),
                -2.0,
            )
            - 0.1
        )
        x_max = (
            max(
                torch.max(x_context).cpu().numpy(),
                torch.max(x_target).cpu().numpy(),
                2.0,
            )
            + 0.1
        )
        num_points = int(
            to_multiple(self.points_per_unit * (x_max - x_min), self.multiplier)
        )
        x_grid = torch.linspace(x_min, x_max, num_points).to(self.device)
        x_grid = x_grid[None, :, None].repeat(x_context.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.encoder(x_context, y_context, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError("Shape changed.")

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_target)
        sigma = self.sigma_layer(x_grid, h, x_target)

        return torch.stack([mean, sigma], dim=-1)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Training Step.

        Args:

        """
        out = self.forward(batch["x_context"], batch["y_context"], batch["x_target"])

        loss = self.loss_fn(out, batch["y_target"])

        self.log("train_loss", loss)  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.train_metrics(self.extract_mean_output(out), batch["y_target"])

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Validation Step.

        Args:

        """
        out = self.forward(batch["x_context"], batch["y_context"], batch["x_target"])

        loss = self.loss_fn(out, batch["y_target"])

        self.log("val_loss", loss)  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.val_metrics(self.extract_mean_output(out), batch["y_target"])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step.

        Args:


        """
        out_dict = self.predict_step(
            batch["x_context"], batch["y_context"], batch["x_target"]
        )
        out_dict["targets"] = batch["y_target"].detach().squeeze(-1).cpu().numpy()

        if batch["inputs"].shape[0] > 1:
            self.test_metrics(
                out_dict["pred"].squeeze(-1), batch["y_target"].squeeze(-1)
            )

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in ["x_context", "y_context", "x_target", "y_target"]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        if "out" in out_dict:
            del out_dict["out"]
        return out_dict

    def predict_step(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> dict[str, ndarray]:
        """Predict Step.

        Args:
            x_context: observation locations of shape [batch, data, features]
            y_context: observation values of shape [batch, data, features]
            x_target: locations of outputs of shape [batch, data, features]
        """
        with torch.no_grad():
            out = self.forward(x_context, y_context, x_target).cpu()

        mean, std = out[..., 0], out[..., 1].numpy()

        quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)

        return {
            "pred": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
            "out": out,
        }
