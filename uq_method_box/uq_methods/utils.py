"""Utilities for UQ-Method Implementations."""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import (
    bnn_conv_layer,
    bnn_linear_layer,
    bnn_lstm_layer,
)

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
    elif loss_fn_name is None:
        return None
    else:
        raise ValueError("Your loss function choice is not supported.")


def merge_list_of_dictionaries(list_of_dicts: List[Dict[str, Any]]):
    """Merge list of dictionaries."""
    merged_dict = defaultdict(list)

    for out in list_of_dicts:
        for k, v in out.items():
            merged_dict[k].extend(v.tolist())

    return merged_dict


def save_predictions_to_csv(outputs: Dict[str, np.ndarray], path: str) -> None:
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


def dnn_to_bnn_some(m, bnn_prior_parameters, num_stochastic_modules: int):
    """Replace linear and conv. layers with stochastic layers.

    Args:
        m: nn.module
        bnn_prior_parameter: dictionary,
            prior_mu: prior mean value for bayesian layer
            prior_sigma: prior variance value for bayesian layer
            posterior_mu_init: mean initialization value for approximate posterior
            posterior_rho_init: variance initialization value for approximate posterior
                through softplus σ = log(1 + exp(ρ))
            bayesian_layer_type: `Flipout` or `Reparameterization
        num_stochastic_modules: number of modules that should be stochastic,
            max value all modules.
    """
    # assert len(list(m.named_modules(remove_duplicate=False)))
    # >= num_stochastic_modules,
    #  "More stochastic modules than modules."

    replace_modules = list(m._modules.items())[-num_stochastic_modules:]

    print(len(list(m._modules.items())))
    print(len(replace_modules))

    for name, value in replace_modules:
        if m._modules[name]._modules:
            dnn_to_bnn_some(
                m._modules[name], bnn_prior_parameters, num_stochastic_modules
            )
        if "Conv" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_conv_layer(bnn_prior_parameters, m._modules[name]))
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_linear_layer(bnn_prior_parameters, m._modules[name]))
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(m, name, bnn_lstm_layer(bnn_prior_parameters, m._modules[name]))
        else:
            pass
    return
