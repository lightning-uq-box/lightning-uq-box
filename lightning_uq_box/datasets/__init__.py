# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""UQ-Regression-Box Datasets."""

from .toy_8_gaussians import Toy8GaussiansDataset
from .toy_donut import ToyDonut
from .toy_gaussian_sideways import ToyGaussianSideWays
from .toy_image_classification import ToyImageClassificationDataset
from .toy_image_regression import ToyImageRegressionDataset
from .toy_image_segmentation import ToySegmentationDataset

__all__ = (
    # Toy Image dataset
    "ToyImageRegressionDataset",
    "ToyImageClassificationDataset",
    "ToySegmentationDataset",
    # Toy 8 Gaussians dataset
    "Toy8GaussiansDataset",
    # Toy Gaussian dataset
    "ToyGaussianSideWays",
    # Toy Donut dataset
    "ToyDonut",
)
