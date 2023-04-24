"""Utilities for UQ-Method Implementations."""

import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import SGD, Adam

from uq_method_box.train_utils import NLL, QuantileLoss


def retrieve_loss_fn(
    loss_fn_name: str, quantiles: Optional[list[float]] = None
) -> nn.Module:
    """Retrieve the desired loss function.

    Args:
        loss_fn_name: name of the loss function, one of ['mse', 'nll', 'quantile']

    Returns
        desired loss function module
    """
    if loss_fn_name == "mse":
        return nn.MSELoss()
    elif loss_fn_name == "nll":
        return NLL()
    elif loss_fn_name == "quantile":
        return QuantileLoss(quantiles)
    elif loss_fn_name is None:
        return None
    else:
        raise ValueError("Your loss function choice is not supported.")


def retrieve_optimizer(optimizer_name: str):
    """Retrieve an optimizer."""
    if optimizer_name == "sgd":
        return SGD
    elif optimizer_name == "adam":
        return Adam


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
