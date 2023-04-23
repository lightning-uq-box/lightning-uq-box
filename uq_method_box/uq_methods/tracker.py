"""Implementation for Checkpoint Tracking.

These are heavily inspired by https://github.com/GSK-AI/afterglow, however,
the ideas are adopted to be integrated within lightning.
"""

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, RandomSampler

# should have a base class tracker that does the buffer, update state dict


class ModelSnapshotTracker:
    """Base Class for methods that track snapshots of model weight settings.

    Useful for SWAG for example.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def _get_buffer_for_param(self, param_name, buffer_name):
        safe_name = param_name.replace(".", "_")
        return getattr(self.module, f"{safe_name}_{buffer_name}")

    def _set_buffer_for_param(self, param_name, buffer_name, value):
        safe_name = param_name.replace(".", "_")
        setattr(self.module, f"{safe_name}_{buffer_name}", value)

    def _update_tracked_state_dict(self, state_dict: Dict[str, nn.Parameter]):
        # PyTorch uses OrderedDicts for state_dict because they can have
        # attributes. It gives state_dict a _metadata attribute which can
        # affect how the state_dict is loaded. We have to copy this here.
        full_state_dict = OrderedDict({**state_dict, **self._untracked_state_dict()})
        full_state_dict._metadata = getattr(self.module.state_dict(), "_metadata", None)

        self.module.load_state_dict(full_state_dict)

    def _untracked_state_dict(self):
        filtered_state_dict = {}
        tracked_keys = {name for name, _ in self.module.named_parameters()}
        for k, v in self.module.state_dict().items():
            if k not in tracked_keys:
                filtered_state_dict[k] = v
        return filtered_state_dict


class SWAGTracker(ModelSnapshotTracker):
    """
    Models the parameter distribution over the training trajectory as a multivariate
    gaussian in a low-rank space. See SWAG paper: https://arxiv.org/abs/1902.02476.

    Args:
        module: module to enable tracking for.
        max_cols: the posterior covariance matrix is dimensionally reduced to this
            dimensionality. Must be greater than 1.
        update_period_in_iters: how often to observe the parameters, in interations
        dataloader_for_batchnorm: if this is is provided, we update the model's
            batchnorm running means and variances every time we sample a new set of
            parameters using the data in the dataloader. This is slow but can improve
            performance significantly. See SWAG paper, and
            :code:`torch.optim.swa_utils.update_bn`. Note that the
            assumptions made about what iterating over the dataloader returns are
            the same as those in :code:`torch.optim.swa_utils.update_bn`: it's
            assumed that iterating produces a sequence of (input_batch, label_batch)
            tuples.
        num_datapoints_for_bn_update: Number of training example to use to perfom the
            batchnorm update.
            If :code:`None`, we use the whole dataset, as in the original SWAG
            paper. It's better to better to set this value to 1 and increase the
            number of SWAG samples drawn when predicting in online mode
            (one example at a time) rather than in batch mode.
            If this is not None, dataloader_for_batchnorm must be
            initialised with :code:`shuffle=True`
    """

    def __init__(
        self,
        module: nn.Module,
        max_cols: int,
        update_period_in_iters: int,
        dataloader_for_batchnorm: Optional[DataLoader] = None,
        num_datapoints_for_bn_update: Optional[int] = None,
    ):
        super().__init__(module)
        self.iterations = 0
        self.update_period_in_iters = update_period_in_iters
        self.max_cols = max_cols

        # the iterations should be steered by the lightning module
        # but this would also require the use of a trainer? Makes it less flexible,
        # or manually call the update method based on the iteration
        # then it can be used with and without a trainer
        self.dataloader_for_batchnorm = dataloader_for_batchnorm

        self.num_datapoints_for_bn_update = num_datapoints_for_bn_update

    def _get_buffer_for_param(self, param_name, buffer_name):
        safe_name = param_name.replace(".", "_")
        return getattr(self.module, f"{safe_name}_{buffer_name}")

    def _set_buffer_for_param(self, param_name, buffer_name, value):
        safe_name = param_name.replace(".", "_")
        setattr(self.module, f"{safe_name}_{buffer_name}", value)

    def _update_tracked_state_dict(self, state_dict: Dict[str, nn.Parameter]):
        # PyTorch uses OrderedDicts for state_dict because they can have
        # attributes. It gives state_dict a _metadata attribute which can
        # affect how the state_dict is loaded. We have to copy this here.
        full_state_dict = OrderedDict({**state_dict, **self._untracked_state_dict()})
        full_state_dict._metadata = getattr(self.module.state_dict(), "_metadata", None)

        self.module.load_state_dict(full_state_dict)

    def _bn_loader_does_not_shuffle(self):
        return hasattr(self.dataloader_for_batchnorm, "sampler") and isinstance(
            self.dataloader_for_batchnorm, RandomSampler
        )

    def _sample_state_dict(self) -> dict:
        if self.module.num_snapshots_tracked == 0:
            raise RuntimeError(
                "Attempted to sample weights using a tracker that has "
                "recorded no snapshots"
            )

        sampled = {}

        _, first_param = next(iter(self.module.named_parameters()))
        K_sample = (
            Normal(torch.zeros(self.max_cols), torch.ones(self.max_cols))
            .sample()
            .to(first_param.device)  # should have lightning device
        )

        for name, _ in self.module.named_parameters():
            mean = self._get_buffer_for_param(name, "mean")
            squared_mean = self._get_buffer_for_param(name, "squared_mean")
            d_block = self._get_buffer_for_param(name, "D_block")
            p1 = mean
            p2 = Normal(
                torch.zeros_like(mean),
                (0.5 * (squared_mean - mean.pow(2)).clamp(1e-30)).sqrt(),
            ).sample()
            shape = d_block.shape[1:]
            aux = d_block.reshape(self.max_cols, -1)
            p3 = torch.matmul(K_sample, aux).reshape(shape) / math.sqrt(
                2 * (self.max_cols - 1)
            )
            sampled[name] = p1 + p2 + p3
        return sampled

    def update_uncertainty_buffers(self):
        """Update the running average over weights."""
        if self.module.num_snapshots_tracked == 0:
            with torch.no_grad():
                for name, parameter in self.module.named_parameters():
                    mean = self._get_buffer_for_param(name, "mean")
                    squared_mean = self._get_buffer_for_param(name, "squared_mean")
                    self._set_buffer_for_param(name, "mean", mean + parameter)
                    self._set_buffer_for_param(
                        name, "squared_mean", squared_mean + parameter.pow(2)
                    )
        else:
            with torch.no_grad():
                for name, parameter in self.module.named_parameters():
                    mean = self._get_buffer_for_param(name, "mean")
                    squared_mean = self._get_buffer_for_param(name, "squared_mean")
                    d_block = self._get_buffer_for_param(name, "D_block")
                    self._set_buffer_for_param(
                        name,
                        "mean",
                        (self.module.num_snapshots_tracked * mean + parameter)
                        / (self.module.num_snapshots_tracked + 1),
                    )
                    self._set_buffer_for_param(
                        name,
                        "squared_mean",
                        (
                            self.module.num_snapshots_tracked * squared_mean
                            + parameter.pow(2)
                        )
                        / (self.module.num_snapshots_tracked + 1),
                    )
                    d_block = d_block.roll(1, dims=0)
                    d_block[0] = parameter - mean
                    self._set_buffer_for_param(name, "D_block", d_block)

        self.module.num_snapshots_tracked += 1

    def sample_state(self, device: str = "cpu"):
        """Update the state of the tracker's :code:`module` with a sample from
        the estimated distribution over parameters.
        Args:
            device: where to send the data duing batchnorm update. Ignored
                if we don't do batchnorm update.
        """
        sampled_state_dict = self._sample_state_dict()
        self._update_tracked_state_dict(sampled_state_dict)
        if self.dataloader_for_batchnorm is not None:
            tracking_was_enabled = self.module.trajectory_tracking_enabled
            self.module.trajectory_tracking_enabled = False
            update_bn(
                self.dataloader_for_batchnorm,
                self.module,
                device=device,
                num_datapoints=self.num_datapoints_for_bn_update,
            )
            self.module.trajectory_tracking_enabled = tracking_was_enabled

    def save(self, path: Union[str, Path]):
        """Save the uncertainty-enabled model so that it can be
        loaded using :code:`afterglow.load_swag_checkpoint`.
        Args:
            path: where to save the checkpoint
        """
        checkpoint_dict = {
            "state_dict": self.module.state_dict(),
            "max_cols": self.max_cols,
        }
        torch.save(checkpoint_dict, path)


def update_bn(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_datapoints: Optional[int] = None,
):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
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
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
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
