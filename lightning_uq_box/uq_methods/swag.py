#    Copyright 2021 GlaxoSmithKline

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Changes include:
# - integrating the functions into pytorch lightning Lightning Module framework
# - enable selections of stochastic modules

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Stochastic Weight Averaging - Gaussian.

Adapted from https://github.com/GSK-AI/afterglow/blob/master/afterglow/trackers/trackers.py (Apache License 2.0) # noqa: E501
for support of partial stochasticity and integration to lightning.
"""

import math
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from .base import DeterministicModel
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    default_segmentation_metrics,
    map_stochastic_modules,
    process_classification_prediction,
    process_regression_prediction,
    process_segmentation_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


class SWAGBase(DeterministicModel):
    """Stochastic Weight Averaging - Gaussian (SWAG).

    If you use this model in your research, please cite the following paper:

    * https://proceedings.neurips.cc/paper_files/paper/2019/hash/118921efba23fc329e6560b27861f0c2-Abstract.html # noqa: E501
    """

    def __init__(
        self,
        model: nn.Module,
        max_swag_snapshots: int,
        snapshot_freq: int,
        num_mc_samples: int,
        swag_lr: float,
        loss_fn: nn.Module,
        stochastic_module_names: Optional[list[Union[int, str]]] = None,
        num_datapoints_for_bn_update: int = 0,
    ) -> None:
        """Initialize a new instance of SWAG Model Wrapper.

        Args:
            model: pytorch model
            max_swag_snapshots: maximum number of snapshots to store
            snapshot_freq: frequency of snapshots
            num_mc_samples: number of MC samples during prediction
            swag_lr: learning rate for swag
            loss_fn: loss function
            stochastic_module_names: list of module names or indices that should
                be converted to variational layers
            num_datapoints_for_bn_update: num of datapoints to use for batchnorm update
        """
        super().__init__(model, loss_fn, None, None)
        self.stochastic_module_names = map_stochastic_modules(
            self.model, stochastic_module_names
        )
        self.swag_fitted = False
        self.current_iteration = 0
        self.num_tracked = 0

        self.model_w_and_b_module_names = self._find_weights_and_bias_modules(
            self.model
        )
        self.max_swag_snapshots = max_swag_snapshots
        self._create_swag_buffers(self.model)

        # manual optimization with SWAG optimization process
        self.automatic_optimization = False

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        pass

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute SWAG optimization step.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader
        """
        swag_opt = self.optimizers()
        swag_opt.zero_grad()

        if self.trainer.global_step % self.hparams.snapshot_freq == 0:
            self.update_uncertainty_buffers()

        loss = self.loss_fn(self.model(batch[self.input_key]), batch[self.target_key])
        self.manual_backward(loss)

        swag_opt.step()

    def on_train_epoch_end(self):
        """Do not Log epoch-level training metrics."""
        pass

    def on_train_end(self) -> None:
        """After training stage is completed, swag is fitted."""
        self.swag_fitted = True

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Not intended to be used."""
        pass

    def on_validation_epoch_end(self) -> None:
        """Do not log any validation metrics."""
        pass

    def _find_weights_and_bias_modules(self, instance: nn.Module) -> list[str]:
        """Find weights and bias modules corresponding to part stochastic modules."""
        model_w_and_b_module_names: list[str] = []
        for name, _ in instance.named_parameters():
            if (
                name.removesuffix(".weight").removesuffix(".bias")
                in self.stochastic_module_names
            ):  # noqa: E501
                model_w_and_b_module_names.append(name)
        return model_w_and_b_module_names

    def _create_swag_buffers(self, instance: nn.Module) -> None:
        """Create swawg buffers for an underlying module.

        Args:
            instance: underlying model instance for which to create buffers
        """
        for name, parameter in instance.named_parameters():
            # check for partial stochasticity modules
            if name in self.model_w_and_b_module_names:
                name = name.replace(".", "_")
                instance.register_buffer(f"{name}_mean", deepcopy(parameter))
                instance.register_buffer(
                    f"{name}_squared_mean", torch.zeros_like(parameter)
                )
                instance.register_buffer(
                    f"{name}_D_block",
                    torch.zeros(
                        (self.max_swag_snapshots, *parameter.shape),
                        device=parameter.device,
                    ),
                )
            else:
                continue
        instance.register_buffer("num_snapshots_tracked", torch.tensor(0, dtype=int))

    def _get_buffer_for_param(self, param_name: str, buffer_name: str):
        """Get buffer for parameter name.

        Args:
            param_name: parameter name
            buffer_name: buffer_name
        """
        safe_name = param_name.replace(".", "_")
        # TODO be able to access and retrieve nested
        # param names in custom models
        return getattr(self.model, f"{safe_name}_{buffer_name}")

    def _set_buffer_for_param(self, param_name, buffer_name, value):
        safe_name = param_name.replace(".", "_")
        setattr(self.model, f"{safe_name}_{buffer_name}", value)

    def _update_tracked_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Update tracked state_dict.

        Args:
            state_dict: model state_dict

        Returns:
            state_dict
        """
        full_state_dict = OrderedDict({**state_dict, **self._untracked_state_dict()})
        full_state_dict._metadata = getattr(self.model.state_dict(), "_metadata", None)

        self.model.load_state_dict(full_state_dict)

    def _untracked_state_dict(self) -> dict[str, nn.Parameter]:
        """Return filtered untracked state dict."""
        filtered_state_dict = {}
        for k, v in self.model.state_dict().items():
            if k not in self.model_w_and_b_module_names:
                filtered_state_dict[k] = v
        return filtered_state_dict

    def _sample_state_dict(self) -> dict:
        """Sample the underlying model state dict."""
        if self.num_tracked == 0:
            raise RuntimeError(
                "Attempted to sample weights using a tracker that has "
                "recorded no snapshots"
            )

        sampled = {}

        # find first param
        for name, param in self.model.named_parameters():
            if name in self.model_w_and_b_module_names:
                K_sample = (
                    Normal(
                        torch.zeros(self.hparams.max_swag_snapshots),
                        torch.ones(self.hparams.max_swag_snapshots),
                    )
                    .sample()
                    .to(param.device)  # should have lightning device
                )
                break
            else:
                continue

        for name in self.model_w_and_b_module_names:
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
                    if name in self.model_w_and_b_module_names:
                        mean = self._get_buffer_for_param(name, "mean")
                        squared_mean = self._get_buffer_for_param(name, "squared_mean")
                        self._set_buffer_for_param(name, "mean", mean + parameter)
                        self._set_buffer_for_param(
                            name, "squared_mean", squared_mean + parameter.pow(2)
                        )
                    else:
                        continue
        else:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.model_w_and_b_module_names:
                        mean = self._get_buffer_for_param(name, "mean")
                        squared_mean = self._get_buffer_for_param(name, "squared_mean")
                        d_block = self._get_buffer_for_param(name, "D_block")
                        self._set_buffer_for_param(
                            name,
                            "mean",
                            (self.num_tracked * mean + parameter)
                            / (self.num_tracked + 1),
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
                    else:
                        continue

        self.num_tracked += 1

    def sample_state(self):
        """Update the state with a sample."""
        sampled_state_dict = self._sample_state_dict()
        self._update_tracked_state_dict(sampled_state_dict)
        if self.hparams.num_datapoints_for_bn_update > 0:
            update_bn(
                self.train_loader,
                self.model,
                device=self.device,
                num_datapoints=self.hparams.num_datapoints_for_bn_update,
            )

    def sample_predictions(self, X: Tensor) -> Tensor:
        """Sample predictions.

        Args:
            X: input batch of shape [batch_size x input_dims]

        Returns:
            predictions of shape [batch_size x num_outputs x num_mc_samples]
        """
        preds = []
        for i in range(self.hparams.num_mc_samples):
            # sample weights
            self.sample_state()
            with torch.no_grad():
                pred = self.model(X)
            preds.append(pred)

        preds = torch.stack(preds, dim=-1)

        return preds

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Manually implemented SWAG optimization."""
        swag_params: list[nn.Parameter] = [
            param
            for name, param in self.model.named_parameters()
            if name in self.model_w_and_b_module_names
        ]
        swag_optimizer = torch.optim.SGD(swag_params, lr=self.hparams.swag_lr)
        return swag_optimizer


class SWAGRegression(SWAGBase):
    """SWAG Model for Regression.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.neurips.cc/paper_files/paper/2019/hash/118921efba23fc329e6560b27861f0c2-Abstract.html # noqa: E501
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        max_swag_snapshots: int,
        snapshot_freq: int,
        num_mc_samples: int,
        swag_lr: float,
        loss_fn: nn.Module,
        stochastic_module_names: Optional[Union[list[int], list[str]]] = None,
        num_datapoints_for_bn_update: int = 0,
    ) -> None:
        """Initialize a new instance of SWAG Model for Regression.

        Args:
            model: pytorch model
            num_swag_epochs: number of epochs to train swag
            max_swag_snapshots: maximum number of snapshots to store
            snapshot_freq: frequency of snapshots
            num_mc_samples: number of MC samples during prediction
            swag_lr: learning rate for swag
            loss_fn: loss function
            stochastic_module_names: names of modules that are partially stochastic
            num_datapoints_for_bn_update: num of datapoints to use for batchnorm update
        """
        super().__init__(
            model,
            max_swag_snapshots,
            snapshot_freq,
            num_mc_samples,
            swag_lr,
            loss_fn,
            stochastic_module_names,
            num_datapoints_for_bn_update,
        )
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_regression_metrics("test")

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],  # type: ignore[override]
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        """Prediction step that with SWAG uncertainty.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            prediction dictionary
        """
        if not self.swag_fitted:
            raise RuntimeError(
                "SWAG is not fitted yet, please call trainer.fit() first."
            )

        preds = self.sample_predictions(X)

        return process_regression_prediction(preds)


class SWAGClassification(SWAGBase):
    """SWAG Model for Classification.

    If you use this model in your research, please cite the following paper:

    * https://proceedings.neurips.cc/paper_files/paper/2019/hash/118921efba23fc329e6560b27861f0c2-Abstract.html # noqa: E501
    """

    pred_file_name = "preds.csv"
    valid_tasks = ["binary", "multiclass", "multilable"]

    def __init__(
        self,
        model: nn.Module,
        max_swag_snapshots: int,
        snapshot_freq: int,
        num_mc_samples: int,
        swag_lr: float,
        loss_fn: nn.Module,
        task: str = "multiclass",
        stochastic_module_names: Optional[Union[list[int], list[str]]] = None,
        num_datapoints_for_bn_update: int = 0,
    ) -> None:
        """Initialize a new instance of SWAG Model for Classification.

        Args:
            model: pytorch model
            num_swag_epochs: number of epochs to train swag
            max_swag_snapshots: maximum number of snapshots to store
            snapshot_freq: frequency of snapshots
            num_mc_samples: number of MC samples during prediction
            swag_lr: learning rate for swag
            loss_fn: loss function
            task: classification task, one of ['binary', 'multiclass', 'multilabel']
            stochastic_module_names: names of modules that are partially stochastic
            num_datapoints_for_bn_update: num of datapoints to use for batchnorm update
        """
        assert task in self.valid_tasks
        self.task = task
        self.num_classes = _get_num_outputs(model)

        super().__init__(
            model,
            max_swag_snapshots,
            snapshot_freq,
            num_mc_samples,
            swag_lr,
            loss_fn,
            stochastic_module_names,
            num_datapoints_for_bn_update,
        )
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        """Prediction step with SWAG uncertainty.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.swag_fitted:
            raise RuntimeError(
                "SWAG is not fitted yet, please call trainer.fit() first."
            )

        preds = self.sample_predictions(X)

        return process_classification_prediction(preds)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],  # type: ignore[override]
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class SWAGSegmentation(SWAGClassification):
    """SWAG Model for Segmentation."""

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.test_metrics = default_segmentation_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        """Prediction step with SWAG uncertainty.

        Args:
            X: prediction batch of shape [batch_size x num_channels x height x width]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            prediction dictionary
        """
        if not self.swag_fitted:
            raise RuntimeError(
                "SWAG is not fitted yet, please call trainer.fit() first."
            )
        preds = self.sample_predictions(X)
        return process_segmentation_prediction(preds)

    def on_test_batch_end(
        self,
        outputs: dict[str, Tensor],  # type: ignore[override]
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pass


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
    for batch in loader:
        if datapoints_used_for_update == num_datapoints:
            break
        if isinstance(batch, (list, tuple)):
            input = batch[0]
        if isinstance(batch, (dict)):
            import pdb

            pdb.set_trace()
            input = batch["image"]
        if device is not None:
            input = input.to(device)
        input = input[: num_datapoints - datapoints_used_for_update]

        model(input)

        datapoints_used_for_update += len(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
