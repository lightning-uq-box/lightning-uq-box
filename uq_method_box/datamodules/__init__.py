"""UQ-Regression-Box Datamodules."""

from .toy_bimodal import ToyBimodalDatamodule
from .toy_heteroscedastic import ToyHeteroscedasticDatamodule
from .toy_image_regression import ToyImageRegressionDatamodule
from .toy_sine import ToySineDatamodule
from .toy_uncertainty_gaps import ToyUncertaintyGaps
from .uci import UCIRegressionDatamodule

__all__ = (
    # toy datamodules
    "ToyBimodalDatamodule",
    "ToySineDatamodule",
    "ToyHeteroscedasticDatamodule",
    "ToyImageRegressionDatamodule",
    "ToyUncertaintyGaps",
    # UCI Data module
    "UCIRegressionDatamodule",
)
