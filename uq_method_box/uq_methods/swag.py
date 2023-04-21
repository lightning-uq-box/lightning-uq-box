"""Stochastic Weight Averaging - Gaussian.

Adapted from https://github.com/GSK-AI/afterglow/blob/master/afterglow/trackers/trackers.py # noqa: E501
"""

import math
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import trange

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .utils import retrieve_loss_fn, save_predictions_to_csv


class SWAGModel(LightningModule):
    """Stochastic Weight Averaging - Gaussian (SWAG)."""

    def __init__(
        self,
        model: nn.Module,
        num_swag_epochs: int,
        max_swag_snapshots: int,
        snapshot_freq: int,
        num_mc_samples: int,
        swag_lr: float,
        loss_fn: str,
        train_loader: DataLoader,
        save_dir: str,
        num_datapoints_for_bn_update: Optional[int] = None,
        target_scaler: StandardScaler = None,
        swag_fitted: bool = False,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            model: lightning module to use as underlying model
            swag_args: laplace arguments to initialize a Laplace Model
            train_loader: train loader to be used but maybe this can
                also be accessed through the trainer or write a
                train_dataloader() method for this model based on the config?
        """
        super().__init__()

        self.save_hyperparameters(ignore=["model", "train_loader"])

        self.train_loader = train_loader
        self.model = model

        self.criterion = retrieve_loss_fn(loss_fn)

        self.current_iteration = 0
        self.num_tracked = 0
        self.target_scaler = target_scaler

        self._create_swag_buffers(model)

    def forward(self, X: Tensor, **kwargs: Any) -> Any:
        """Forward method."""
        if self.hparams.swag_fitted:
            return self.model(X)
        else:
            raise RuntimeError("Forward only after SWAG has been fitted.")

    def _create_swag_buffers(self, instance: nn.Module) -> None:
        """Create swawg buffers for an underlying module.

        Args:
            instance: underlying model instance for which to create buffers
        """
        for name, parameter in instance.named_parameters():
            name = name.replace(".", "_")
            instance.register_buffer(f"{name}_mean", deepcopy(parameter))
            instance.register_buffer(
                f"{name}_squared_mean", torch.zeros_like(parameter)
            )
            instance.register_buffer(
                f"{name}_D_block",
                torch.zeros(
                    (self.hparams.max_swag_snapshots, *parameter.shape),
                    device=parameter.device,
                ),
            )
        instance.register_buffer("num_snapshots_tracked", torch.tensor(0, dtype=int))
        # return instance

    def _get_buffer_for_param(self, param_name: str, buffer_name: str):
        """Get buffer for parameter name.

        Args:
            param_name: parameter name
            buffer_name: buffer_name
        """
        safe_name = param_name.replace(".", "_")
        return getattr(self.model, f"{safe_name}_{buffer_name}")

    def _set_buffer_for_param(self, param_name, buffer_name, value):
        safe_name = param_name.replace(".", "_")
        setattr(self.model, f"{safe_name}_{buffer_name}", value)

    def _update_tracked_state_dict(self, state_dict: Dict[str, nn.Parameter]) -> None:
        """Update tracked state_dict.

        Args:
            state_dict: model state_dict

        Returns:
            state_dict
        """
        # PyTorch uses OrderedDicts for state_dict because they can have
        # attributes. It gives state_dict a _metadata attribute which can
        # affect how the state_dict is loaded. We have to copy this here.
        full_state_dict = OrderedDict({**state_dict, **self._untracked_state_dict()})
        full_state_dict._metadata = getattr(self.model.state_dict(), "_metadata", None)

        self.model.load_state_dict(full_state_dict)

    def _untracked_state_dict(self) -> Dict[str, nn.Parameter]:
        """Return filtered untracked state dict."""
        filtered_state_dict = {}
        tracked_keys = {name for name, _ in self.model.named_parameters()}
        for k, v in self.model.state_dict().items():
            if k not in tracked_keys:
                filtered_state_dict[k] = v
        return filtered_state_dict

    def _sample_state_dict(self) -> dict:
        """Sample the underlying model state dict."""
        if not self.hparams.swag_fitted:
            raise RuntimeError(
                "Attempted to sample weights using a tracker that has "
                "recorded no snapshots"
            )

        sampled = {}

        _, first_param = next(iter(self.model.named_parameters()))
        K_sample = (
            Normal(
                torch.zeros(self.hparams.max_swag_snapshots),
                torch.ones(self.hparams.max_swag_snapshots),
            )
            .sample()
            .to(first_param.device)  # should have lightning device
        )

        for name, _ in self.model.named_parameters():
            mean = self._get_buffer_for_param(name, "mean")
            squared_mean = self._get_buffer_for_param(name, "squared_mean")
            d_block = self._get_buffer_for_param(name, "D_block")
            p1 = mean
            p2 = Normal(
                torch.zeros_like(mean),
                (0.5 * (squared_mean - mean.pow(2)).clamp(1e-30)).sqrt(),
            ).sample()
            shape = d_block.shape[1:]
            aux = d_block.reshape(self.hparams.max_swag_snapshots, -1)
            p3 = torch.matmul(K_sample, aux).reshape(shape) / math.sqrt(
                2 * (self.hparams.max_swag_snapshots - 1)
            )
            sampled[name] = p1 + p2 + p3
        return sampled

    def update_uncertainty_buffers(self):
        """Update the running average over weights."""
        if self.num_tracked == 0:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    mean = self._get_buffer_for_param(name, "mean")
                    squared_mean = self._get_buffer_for_param(name, "squared_mean")
                    self._set_buffer_for_param(name, "mean", mean + parameter)
                    self._set_buffer_for_param(
                        name, "squared_mean", squared_mean + parameter.pow(2)
                    )
        else:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    mean = self._get_buffer_for_param(name, "mean")
                    squared_mean = self._get_buffer_for_param(name, "squared_mean")
                    d_block = self._get_buffer_for_param(name, "D_block")
                    self._set_buffer_for_param(
                        name,
                        "mean",
                        (self.num_tracked * mean + parameter) / (self.num_tracked + 1),
                    )
                    self._set_buffer_for_param(
                        name,
                        "squared_mean",
                        (self.num_tracked * squared_mean + parameter.pow(2))
                        / (self.num_tracked + 1),
                    )
                    d_block = d_block.roll(1, dims=0)
                    d_block[0] = parameter - mean
                    self._set_buffer_for_param(name, "D_block", d_block)

        self.num_tracked += 1

    def sample_state(self):
        """Update the state with a sample."""
        sampled_state_dict = self._sample_state_dict()
        self._update_tracked_state_dict(sampled_state_dict)
        if self.train_loader is not None:
            # tracking_was_enabled = self.model.trajectory_tracking_enabled
            # self.model.trajectory_tracking_enabled = False
            update_bn(
                self.train_loader,
                self.model,
                device=self.device,
                num_datapoints=self.hparams.num_datapoints_for_bn_update,
            )
            # self.model.trajectory_tracking_enabled = tracking_was_enabled

    def on_test_start(self) -> None:
        """Fit the SWAG approximation."""
        if not self.hparams.swag_fitted:
            swag_optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.hparams.swag_lr
            )

            # lightning automatically disables gradient computation during test
            with torch.inference_mode(False):
                bar = trange(self.hparams.num_swag_epochs)
                # num epochs
                for i in bar:
                    for X, y in self.train_loader:
                        if self.current_iteration % self.hparams.snapshot_freq == 0:
                            self.update_uncertainty_buffers()

                        self.current_iteration += 1

                        # do model forward pass and sgd update
                        swag_optimizer.zero_grad()
                        out = self.model(X)
                        loss = self.criterion(out, y)
                        loss.backward()
                        swag_optimizer.step()

            self.hparams["swag_fitted"] = True

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Test step."""
        X, y = args[0]
        out_dict = self.predict_step(X)
        out_dict["targets"] = y.squeeze(-1).numpy()
        return out_dict

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

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Prediction step that produces conformalized prediction sets.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.hparams.swag_fitted:
            self.on_test_start()

        preds = []
        for i in range(self.hparams.num_mc_samples):
            # sample weights
            self.sample_state()
            with torch.no_grad():
                pred = self.forward(X).cpu().numpy()
            preds.append(pred)

        preds = np.stack(preds, axis=-1)

        mean_samples = preds[:, 0, :]

        # assume prediction with sigma
        # this is also quiet common across models so standardize this
        if preds.shape[1] == 2:
            log_sigma_2_samples = preds[:, 1, :]
            eps = np.ones_like(log_sigma_2_samples) * 1e-6
            sigma_samples = np.sqrt(eps + np.exp(log_sigma_2_samples))
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
        # assume mse prediction
        else:
            mean = mean_samples.mean(-1)
            std = mean_samples.std(-1)
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": std,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }


# Adapted from https://github.com/GSK-AI/afterglow/blob/master/afterglow/trackers/batchnorm.py # noqa: E501
def update_bn(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_datapoints: Optional[int] = None,
):
    """Update BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        loader: dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model: model for which we seek to update BatchNorm
            statistics.
        device: If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
        num_datapoints: number of examples to use to perform the update.

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    if num_datapoints is None:
        num_datapoints = len(loader.dataset)

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    datapoints_used_for_update = 0
    for input in loader:
        if datapoints_used_for_update == num_datapoints:
            break
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)
        input = input[: num_datapoints - datapoints_used_for_update]

        model(input)

        datapoints_used_for_update += len(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
