"""UQ-Regression-Box Datasets."""

# from .reforesTree import ReforesTreeRegression
from .uci import UCIRegressionDataset
from .uci_boston import UCIBoston
from .uci_concrete import UCIConcrete
from .uci_energy import UCIEnergy
from .uci_naval import UCINaval
from .uci_yacht import UCIYacht

__all__ = (
    "UCIRegressionDataset",
    "UCIBoston",
    "UCIEnergy",
    "UCIConcrete",
    "UCINaval",
    "UCIYacht",
)
