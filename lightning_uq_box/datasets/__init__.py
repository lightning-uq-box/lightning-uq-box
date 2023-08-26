"""UQ-Regression-Box Datasets."""

# from .reforesTree import ReforesTreeRegression
from .toy_image_regression import ToyImageRegressionDataset
from .uci import UCIRegressionDataset
from .uci_boston import UCIBoston
from .uci_concrete import UCIConcrete
from .uci_energy import UCIEnergy
from .uci_naval import UCINaval
from .uci_yacht import UCIYacht
from .usa_vars import (
    USAVarsFeatureExtracted,
    USAVarsFeaturesOOD,
    USAVarsFeaturesOur,
    USAVarsOOD,
)

__all__ = (
    # Toy Image dataset
    "ToyImageRegressionDataset",
    # UCI Datasets
    "UCIRegressionDataset",
    "UCIBoston",
    "UCIEnergy",
    "UCIConcrete",
    "UCINaval",
    "UCIYacht",
    # Image Datasets
    "USAVarsOOD",
    "USAVarsFeaturesOOD",
    "USAVarsFeatureExtracted",
    "USAVarsFeaturesOur",
)
