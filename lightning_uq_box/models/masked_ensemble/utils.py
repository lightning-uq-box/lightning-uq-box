# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Masked Ensemble Model Utilities to convert model to Masked Ensemble."""

import torch.nn as nn

from .masked_layers import MaskedConv2d, MaskedLinear


def convert_deterministic_to_masked_ensemble(
    deterministic_model: nn.Module, num_estimators: int, scale: float
) -> None:
    """Convert a deterministic model to a Masked Ensemble model.

    Converts layers that have more than 10 input features or channels to
    a Masked Ensemble layer.

    Args:
        deterministic_model: PyTorch model to turn into a Masked Ensemble.
        num_estimators: The number of estimators (masks) to generate.
        scale: The scale factor for mask generation. Muste be a scaler in
            the interval [1, 6].
    """
    for name, value in list(deterministic_model._modules.items()):
        if deterministic_model._modules[name]._modules:
            convert_deterministic_to_masked_ensemble(
                deterministic_model._modules[name],
                num_estimators=num_estimators,
                scale=scale,
            )
        elif "Conv2d" in deterministic_model._modules[name].__class__.__name__:
            curr_layer = deterministic_model._modules[name]
            if curr_layer.in_channels >= 10:
                setattr(
                    deterministic_model,
                    name,
                    MaskedConv2d(
                        num_estimators=num_estimators,
                        scale=scale,
                        in_channels=curr_layer.in_channels,
                        out_channels=curr_layer.out_channels,
                        kernel_size=curr_layer.kernel_size,
                        stride=curr_layer.stride,
                        padding=curr_layer.padding,
                        dilation=curr_layer.dilation,
                        groups=curr_layer.groups,
                        bias=curr_layer.bias is not None,
                    ),
                )

        elif "Linear" in deterministic_model._modules[name].__class__.__name__:
            curr_layer = deterministic_model._modules[name]
            if curr_layer.in_features >= 10:
                setattr(
                    deterministic_model,
                    name,
                    MaskedLinear(
                        num_estimators=num_estimators,
                        scale=scale,
                        in_features=curr_layer.in_features,
                        out_features=curr_layer.out_features,
                        bias=curr_layer.bias is not None,
                    ),
                )
        else:
            pass
