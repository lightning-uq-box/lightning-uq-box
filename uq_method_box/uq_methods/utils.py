"""Utilities for UQ-Method Implementations."""

import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from bayesian_torch.models.dnn_to_bnn import (
    bnn_conv_layer,
    bnn_linear_layer,
    bnn_lstm_layer,
)
from torch import Tensor
from torch.optim import SGD, Adam

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .loss_functions import NLL, QuantileLoss


def process_model_prediction(
    preds: Tensor, quantiles: list[float]
) -> dict[str, np.ndarray]:
    """Process model predictions that could be mse or nll predictions.

    Args:
        preds: prediction tensor of shape [batch_size, num_outputs, num_samples]
        quantiles: quantiles to compute

    Returns:
        dictionary with mean and uncertainty predictions
    """
    mean_samples = preds[:, 0, :]
    # assume nll prediction with sigma
    if preds.shape[1] == 2:
        log_sigma_2_samples = preds[:, 1, :]
        eps = np.ones_like(log_sigma_2_samples) * 1e-6
        sigma_samples = np.sqrt(eps + np.exp(log_sigma_2_samples))
        mean = mean_samples.mean(-1)
        std = compute_predictive_uncertainty(mean_samples, sigma_samples)
        std[std <= 0] = 1e-6
        aleatoric = compute_aleatoric_uncertainty(sigma_samples)
        epistemic = compute_epistemic_uncertainty(mean_samples)
        quantiles = compute_quantiles_from_std(mean, std, quantiles)
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
        std[std <= 0] = 1e-6
        quantiles = compute_quantiles_from_std(mean, std, quantiles)

        return {
            "mean": mean,
            "pred_uct": std,
            "epistemic_uct": std,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }


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
