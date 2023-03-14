"""Utilities for UQ-Method Implementations."""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch.nn as nn

from uq_method_box.train_utils import NLL, QuantileLoss


def retrieve_loss_fn(
    loss_fn_name: str, quantiles: Optional[List[float]] = None
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
    else:
        raise ValueError("Your loss function choice is not supported.")


def merge_list_of_dictionaries(list_of_dicts: List[Dict[str, Any]]):
    """Merge list of dictionaries."""
    merged_dict = defaultdict(list)

    for out in list_of_dicts:
        for k, v in out.items():
            merged_dict[k].extend(v.tolist())

    return merged_dict


def save_predictions_to_csv(outputs: List[Dict[str, np.ndarray]], path: str) -> None:
    """Save model predictions to csv file.

    Args:
        outputs: a NxO array where N is the number of predictions and
            O the number of variables to be stored
        path: path where csv should be saved
    """
    # concatenate the predictions into a single dictionary
    save_pred_dict = merge_list_of_dictionaries(outputs)

    # save the outputs, i.e. write them to file
    df = pd.DataFrame.from_dict(save_pred_dict)

    df.to_csv(path, index=False)
