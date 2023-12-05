# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""UQ-Regression-Box Datamodules."""

from .toy_8_gaussians import Toy8GaussiansDataModule
from .toy_donut import ToyDonutDataModule
from .toy_due import ToyDUE
from .toy_gaussian_sideways import ToyGaussianSideWaysDataModule
from .toy_half_moons import TwoMoonsDataModule
from .toy_heteroscedastic import ToyHeteroscedasticDatamodule
from .toy_image_classification import ToyImageClassificationDatamodule
from .toy_image_regression import ToyImageRegressionDatamodule
from .toy_image_segmentation import ToySegmentationDataModule
from .toy_sine import ToySineDatamodule
from .toy_uncertainty_gaps import ToyUncertaintyGaps

__all__ = (
    # toy datamodules
    "ToyBimodalDatamodule",
    "ToySineDatamodule",
    "TwoMoonsDataModule",
    "ToyHeteroscedasticDatamodule",
    "ToyImageClassificationDatamodule",
    "ToyImageRegressionDatamodule",
    "ToySegmentationDataModule",
    "ToyUncertaintyGaps",
    "ToyDUE",
    "Toy8GaussiansDataModule",
    "ToyGaussianSideWaysDataModule",
    "ToyDonutDataModule",
)
