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
        assert value is not None
        if value._modules:
            convert_deterministic_to_masked_ensemble(
                value, num_estimators=num_estimators, scale=scale
            )
        elif isinstance(value, nn.Conv2d):
            if value.in_channels >= 10:
                setattr(
                    deterministic_model,
                    name,
                    MaskedConv2d(
                        num_estimators=num_estimators,
                        scale=scale,
                        in_channels=value.in_channels,
                        out_channels=value.out_channels,
                        kernel_size=value.kernel_size,  # type: ignore[arg-type]
                        stride=value.stride,  # type: ignore[arg-type]
                        padding=value.padding,  # type: ignore[arg-type]
                        dilation=value.dilation,  # type: ignore[arg-type]
                        groups=value.groups,
                        bias=value.bias is not None,
                    ),
                )

        elif isinstance(value, nn.Linear):
            if value.in_features >= 10:
                setattr(
                    deterministic_model,
                    name,
                    MaskedLinear(
                        num_estimators=num_estimators,
                        scale=scale,
                        in_features=value.in_features,
                        out_features=value.out_features,
                        bias=value.bias is not None,
                    ),
                )
        else:
            pass
