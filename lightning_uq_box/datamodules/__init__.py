"""UQ-Regression-Box Datamodules."""

from .toy_due import ToyDUE
from .toy_half_moons import TwoMoonsDataModule
from .toy_heteroscedastic import ToyHeteroscedasticDatamodule
from .toy_image_classification import ToyImageClassificationDatamodule
from .toy_image_regression import ToyImageRegressionDatamodule
from .toy_sine import ToySineDatamodule
from .toy_uncertainty_gaps import ToyUncertaintyGaps

__all__ = (
    # toy datamodules
    "ToyBimodalDatamodule",
    "ToySineDatamodule",
    "TwoMoonsDataModule",
    "ToyHeteroscedasticDatamodule",
    "ToyImageRegressionDatamodule",
    "ToyImageClassificationDatamodule",
    "ToyUncertaintyGaps",
    "ToyDUE",
)
