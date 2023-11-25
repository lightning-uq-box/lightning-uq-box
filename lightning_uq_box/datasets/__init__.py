"""UQ-Regression-Box Datasets."""

from .toy_image_classification import ToyImageClassificationDataset
from .toy_image_regression import ToyImageRegressionDataset
from .uci import UCIRegressionDataset
from .uci_boston import UCIBoston
from .uci_concrete import UCIConcrete
from .uci_energy import UCIEnergy
from .uci_naval import UCINaval
from .uci_yacht import UCIYacht

__all__ = (
    # Toy Image dataset
    "ToyImageRegressionDataset",
    "ToyImageClassificationDataset",
    # UCI Datasets
    "UCIRegressionDataset",
    "UCIBoston",
    "UCIEnergy",
    "UCIConcrete",
    "UCINaval",
    "UCIYacht",
)
