"""UQ-Regression-Box Datamodules."""

from .toy_sine import ToySineDatamodule
from .uci import UCIRegressionDatamodule

__all__ = (
    # datamodules
    "ToySineDatamodule",
    "UCIRegressionDatamodule",
)
