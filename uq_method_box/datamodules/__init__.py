"""UQ-Regression-Box Datamodules."""

from .toy_due import ToyDUE
from .toy_heteroscedastic import ToyHeteroscedasticDatamodule
from .toy_image_regression import ToyImageRegressionDatamodule
from .toy_sine import ToySineDatamodule
from .toy_uncertainty_gaps import ToyUncertaintyGaps
from .uci import UCIRegressionDatamodule

__all__ = (
    # toy datamodules
    "ToySineDatamodule",
    "ToyHeteroscedasticDatamodule",
    "ToyImageRegressionDatamodule",
    "ToyUncertaintyGaps",
    "ToyDUE",
    # UCI Data module
    "UCIRegressionDatamodule",
)
