"""Utilities for UQ-Method Implementations."""

import os
from collections import OrderedDict, defaultdict
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import (
    Accuracy,
    CalibrationError,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    R2Score,
)

from lightning_uq_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)


def default_regression_metrics(prefix: str):
    """Return a set of default regression metrics."""
    return MetricCollection(
        {
            "RMSE": MeanSquaredError(squared=False),
            "MAE": MeanAbsoluteError(),
            "R2": R2Score(),
        },
        prefix=prefix,
    )


def default_classification_metrics(prefix: str, task: str, num_classes: int):
    """Return a set of default classification metrics."""
    return MetricCollection(
        {
            "Acc": Accuracy(task=task, num_classes=num_classes),
            # "CalibErr": CalibrationError(task),
            "F1Score": F1Score(task, num_classes=num_classes),
        },
        prefix=prefix,
    )


def process_regression_prediction(
    preds: Tensor, quantiles: list[float]
) -> dict[str, np.ndarray]:
    """Process regression predictions that could be mse or nll predictions.

    Args:
        preds: prediction tensor of shape [batch_size, num_outputs, num_samples]
        quantiles: quantiles to compute

    Returns:
        dictionary with mean prediction and predictive uncertainty
    """
    mean_samples = preds[:, 0, :].cpu()
    mean = preds[:, 0:1, :].mean(-1)
    # assume nll prediction with sigma
    if preds.shape[1] == 2:
        log_sigma_2_samples = preds[:, 1, :].cpu()
        eps = torch.ones_like(log_sigma_2_samples) * 1e-6
        sigma_samples = torch.sqrt(eps + torch.exp(log_sigma_2_samples))
        std = compute_predictive_uncertainty(mean_samples, sigma_samples)
        aleatoric = compute_aleatoric_uncertainty(sigma_samples)
        epistemic = compute_epistemic_uncertainty(mean_samples)
        quantiles = compute_quantiles_from_std(
            mean.detach().cpu().numpy(), std, quantiles
        )
        return {
            "pred": mean,
            "pred_uct": std,
            "epistemic_uct": epistemic,
            "aleatoric_uct": aleatoric,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
    # assume mse prediction
    else:
        std = mean_samples.std(-1)
        quantiles = compute_quantiles_from_std(
            mean.detach().cpu().numpy(), std, quantiles
        )

        return {
            "pred": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }


def process_classification_prediction(preds: Tensor) -> dict[str, np.ndarray]:
    """Process classification predictions.

    Applies softmax to logit and computes mean over the samples and entropy.

    Args:
        preds: prediction logits tensor of shape [batch_size, num_classes, num_samples]

    Returns:
        dictionary with mean and predictive uncertainty
    """
    mean = nn.functional.softmax(preds.mean(-1), dim=-1)
    entropy = -(mean * mean.log()).sum(dim=-1)

    return {"pred": mean, "pred_uct": entropy}


def change_inplace_activation(module):
    """Change inplace activation."""
    if hasattr(module, "inplace"):
        module.inplace = False


def merge_list_of_dictionaries(list_of_dicts: list[dict[str, Any]]):
    """Merge list of dictionaries."""
    merged_dict = defaultdict(list)

    for out in list_of_dicts:
        for k, v in out.items():
            merged_dict[k].extend(v.tolist())

    return merged_dict


def save_predictions_to_csv(outputs: dict[str, np.ndarray], path: str) -> None:
    """Save model predictions to csv file.

    Args:
        outputs: metrics and values to be saved
        path: path where csv should be saved
    """
    # concatenate the predictions into a single dictionary
    # save_pred_dict = merge_list_of_dictionaries(outputs)

    # save the outputs, i.e. write them to file
    df = pd.DataFrame.from_dict(outputs)

    # check if path already exists, then just append
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:  # create new csv
        df.to_csv(path, index=False)


def map_stochastic_modules(
    model: nn.Module, part_stoch_module_names: Union[None, list[str, int]]
) -> list[str]:
    """Retrieve desired stochastic module names from user arg.

    Args:
        model: model from which to retrieve the module names
        part_stoch_module_names: argument to uq_method for partial stochasticity

    Returns:
        list of desired partially stochastic module names
    """
    ordered_module_names: list[str] = []
    # ignore batchnorm
    for name, val in model.named_parameters():
        # module = getattr(model, )
        ordered_module_names.append(".".join(name.split(".")[:-1]))
    ordered_module_names = list(OrderedDict.fromkeys(ordered_module_names))

    # split of weight/bias
    ordered_module_params = [
        name for name, val in list(model.named_parameters())
    ]  # all
    module_names = [".".join(name.split(".")[:-1]) for name in ordered_module_params]
    # remove duplicates due to weight/bias
    module_names = list(set(module_names))

    if not part_stoch_module_names:  # None means fully stochastic
        part_stoch_names = module_names.copy()
    elif all(isinstance(elem, int) for elem in part_stoch_module_names):
        part_stoch_names = [
            ordered_module_names[idx] for idx in part_stoch_module_names
        ]  # retrieve last ones
    elif all(isinstance(elem, str) for elem in part_stoch_module_names):
        assert set(part_stoch_module_names).issubset(module_names), (
            f"Model only contains these parameter modules {module_names}, "
            f"and you requested {part_stoch_module_names}."
        )
        part_stoch_names = module_names.copy()
    else:
        raise ValueError
    return part_stoch_names


def _get_input_layer_name_and_module(model: nn.Module) -> tuple[str, nn.Module]:
    """Retrieve the input layer name and module from a timm model.

    Args:
        model: pytorch model

    Returns:
        input key and module
    """
    keys = []
    children = list(model.named_children())
    while children != []:
        name, module = children[0]
        keys.append(name)
        children = list(module.named_children())

    key = ".".join(keys)
    return key, module


def _get_output_layer_name_and_module(model: nn.Module) -> tuple[str, nn.Module]:
    """Retrieve the output layer name and module from a pytorch model.

    Args:
        model: pytorch model

    Returns:
        output key and module
    """
    keys = []
    children = list(model.named_children())
    while children != []:
        name, module = children[-1]
        keys.append(name)
        children = list(module.named_children())

    key = ".".join(keys)

    return key, module


def _get_num_inputs(module):
    """Get the number of inputs for a module."""
    _, module = _get_input_layer_name_and_module(module)
    if hasattr(module, "in_features"):  # Linear Layer
        num_inputs = module.in_features
    elif hasattr(module, "in_channels"):  # Conv Layer
        num_inputs = module.in_channels
    return num_inputs


def _get_num_outputs(module: nn.Module) -> int:
    """Get the number of outputs for a module."""
    _, module = _get_output_layer_name_and_module(module)
    if hasattr(module, "out_features"):  # Linear Layer
        num_outputs = module.out_features
    elif hasattr(module, "out_channels"):  # Conv Layer
        num_outputs = module.out_channels
    return num_outputs
