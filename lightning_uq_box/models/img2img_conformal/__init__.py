# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Img2Img Conformal Models UQ Layers."""

from .quantile_regression import QuantileLoss, QuantileRegressionLayer

__all__ = ("QuantileRegressionLayer", "QuantileLoss")
