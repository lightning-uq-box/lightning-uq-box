"""ReforesTree Datamodule for OOD Experiments."""

from typing import Any

from torchgeo.datamodules import NonGeoDataModule

from lightning_uq_box.datasets import ReforesTreeRegression


class ReforesTreeDataModule(NonGeoDataModule):
    """ReforesTree Data Module."""

    def __init__(
        self, batch_size: int = 1, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new instance of Data Module."""
        datatest = ReforesTreeRegression()
        super().__init__(datatest, batch_size, num_workers, **kwargs)
