"""UQ-Regression-Box Datamodules."""

from .toy_heteroscedastic import ToyHeteroscedasticDatamodule
from .toy_image_regression import ToyImageRegressionDatamodule
from .toy_sine import ToySineDatamodule
from .uci import UCIRegressionDatamodule

__all__ = (
    # toy datamodules
    "ToySineDatamodule",
    "ToyHeteroscedasticDatamodule",
    "ToyImageRegressionDatamodule",
    # UCI Data module
    "UCIRegressionDatamodule",
)
