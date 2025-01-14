# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted for Lightning from https://github.com/tonyduan/mixture-density-network
# which is under MIT-License.

"""Mixture Density Networks for Regression."""

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.optim.adam import Adam as Adam

from ..models.mixture_density import MixtureDensityLayer
from .base import DeterministicRegression
from .loss_functions import MixtureDensityLoss
from .utils import _get_output_layer_name_and_module, replace_module


class MDNRegression(DeterministicRegression):
    """Mixture Density Network for Regression.

    The implementation replaces the last linear layer of a
    neural network with a Mixture Density Layer.

    If you use this model in your research, please cite:

    * https://publications.aston.ac.uk/id/eprint/373/
    """

    valid_noise_types = ("diagonal", "isotropic", "isotropic_clusters", "fixed")

    def __init__(
        self,
        model: nn.Module,
        n_components: int,
        hidden_dims: list[int] = [50],
        noise_type: str = "diagonal",
        fixed_noise_level: float | None = None,
        loss_fn: nn.Module | None = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize a new Mixture Density Network.

        Args:
            model: backbone pytorch model
            n_components: number of mixture components
            hidden_dims: hidden dimensionality of :class:`..models.mixture_density.MixtureDensityLayer`
            noise_type: type of noise to model, choose one of
                ('diagonal', 'isotropic', 'isotropic_clusters', 'fixed')
            fixed_noise_level: in case of 'fixed' noise_type, specify the fixed noise
                level you want to use
            loss_fn: Optional loss function, by default
                :class:`.loss_functions.MixtureDensityLoss` will be used
            freeze_backbone: If True, all model parameters except the last
                Mixture Density Layer are frozen
            optimizer: The optimizer to use for training
            lr_scheduler: The learning rate scheduler to use for training

        """
        assert noise_type in self.valid_noise_types, (
            f"Please choose one of {self.valid_noise_types}, you specified {noise_type}."
        )

        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.noise_type = noise_type
        self.fixed_noise_level = fixed_noise_level

        if loss_fn is None:
            loss_fn = MixtureDensityLoss()

        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

        self.freeze_backbone = freeze_backbone
        self._build_model()
        self.freeze_model()

    def _build_model(self) -> None:
        """Build the Mixture Density Network model."""
        last_layer_name, last_module_backbone = _get_output_layer_name_and_module(
            self.model
        )

        in_features = last_module_backbone.in_features
        num_targets = last_module_backbone.out_features

        mdn_layer = MixtureDensityLayer(
            dim_in=in_features,
            dim_out=num_targets,
            hidden_dims=self.hidden_dims,
            n_components=self.n_components,
            noise_type=self.noise_type,
            fixed_noise_level=self.fixed_noise_level,
        )

        replace_module(self.model, last_layer_name, mdn_layer)

    def freeze_model(self) -> None:
        """Freeze model."""
        if self.freeze_backbone:
            # Freeze all parameters initially
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only the parameters of the MixtureDensityLayer
            for _, module in self.model.named_modules():
                if module.__class__.__name__ == "MixtureDensityLayer":
                    for param in module.parameters():
                        param.requires_grad = True

    def adapt_output_for_metrics(self, mu: Tensor, indices: Tensor) -> Tensor:
        """Adapt the output for metrics.

        Args:
            mu: the mu output from the MDN module
            indices: the indices to select

        Returns:
            the mean prediction
        """
        return torch.take_along_dim(mu, indices=indices, dim=1).squeeze(dim=1)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            training loss
        """
        log_pi, mu, sigma = self.model.forward(batch[self.input_key])
        loss = self.loss_fn(log_pi, mu, sigma, batch[self.target_key])

        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(
                self.adapt_output_for_metrics(
                    mu,
                    self.select_mixture_component_preds(batch[self.input_key], log_pi),
                ),
                batch[self.target_key],
            )

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            val loss
        """
        log_pi, mu, sigma = self.model.forward(batch[self.input_key])

        loss = self.loss_fn(log_pi, mu, sigma, batch[self.target_key])

        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(
                self.adapt_output_for_metrics(
                    mu,
                    self.select_mixture_component_preds(batch[self.input_key], log_pi),
                ),
                batch[self.target_key],
            )

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            test loss
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key]

        test_loss = self.loss_fn(
            pred_dict["log_pi"],
            pred_dict["mu"],
            pred_dict["sigma"],
            batch[self.target_key],
        )

        self.log("test_loss", test_loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(
                    pred_dict["mu"],
                    self.select_mixture_component_preds(
                        batch[self.input_key], pred_dict["log_pi"]
                    ),
                ),
                batch[self.target_key],
            )

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)

        del pred_dict["log_pi"]
        del pred_dict["mu"]
        del pred_dict["sigma"]
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction Step with MDN model.

        Args:
            X: input Tensor for prediction
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        Returns:
            prediction dictionary
        """
        with torch.no_grad():
            log_pi, mu, sigma = self.model.forward(X)
            indices = self.select_mixture_component_preds(X, log_pi)
            pred = torch.take_along_dim(mu, indices=indices, dim=1).squeeze(dim=1)
            pred_uct = torch.take_along_dim(sigma, indices=indices, dim=1).squeeze(
                dim=1
            )

        return {
            "pred": pred,
            "pred_uct": pred_uct,
            "log_pi": log_pi,
            "mu": mu,
            "sigma": sigma,
        }

    def select_mixture_component_preds(self, X: Tensor, log_pi: Tensor) -> Tensor:
        """Compute indices to select from mixture components.

        Args:
            X: input tensor of shape [batch_size, input_dim]
            log_pi: of shape [batch_size, n_components]

        Returns:
            indices that can be used for sampling or selection
            of mixture components
        """
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(X), 1).to(X)
        rand_pi = torch.searchsorted(cum_pi, rvs).unsqueeze(-1)
        return rand_pi

    def sample(self, X: Tensor) -> Tensor:
        """Sample from the Mixture Density Network, conditioned on the input.

        Args:
            X: input Tensor of shape [batch_size, dim_in]

        Returns:
            samples of dimension [batch_size, n_components, dim_out]
        """
        with torch.no_grad():
            log_pi, mu, sigma = self.model.forward(X)
        rand_pi = self.select_mixture_component_preds(X, log_pi)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi, dim=1).squeeze(
            dim=1
        )
        return samples
