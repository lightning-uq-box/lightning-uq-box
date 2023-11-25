"""UQ-Regression-Box Datasets."""

from .toy_image_classification import ToyImageClassificationDataset
from .toy_image_regression import ToyImageRegressionDataset

__all__ = (
    # Toy Image dataset
    "ToyImageRegressionDataset",
    "ToyImageClassificationDataset",
)
