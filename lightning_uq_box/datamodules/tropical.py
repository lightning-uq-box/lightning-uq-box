"""Tropical Cyclone Wind Speed Estimation adaptation for OOD experiments."""

from typing import Any

from torchgeo.datamodules import TropicalCycloneDataModule


class TropicalCycloneDataModuleOOD(TropicalCycloneDataModule):
    """Adaptation for Data Module for OOD Experiments.

    Wrapper around TorchGeo Datamodule.

    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new instance of Data Module."""
        super().__init__(batch_size, num_workers, **kwargs)
