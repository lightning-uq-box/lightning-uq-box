# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""UQ-Regression-Box Datasets."""

from .toy_image_classification import ToyImageClassificationDataset
from .toy_image_regression import ToyImageRegressionDataset

__all__ = (
    # Toy Image dataset
    "ToyImageRegressionDataset",
    "ToyImageClassificationDataset",
)
