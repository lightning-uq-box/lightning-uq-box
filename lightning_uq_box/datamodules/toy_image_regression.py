"""Toy Image Regression Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageRegressionDataset


class ToyImageRegressionDatamodule(LightningDataModule):
    """Toy Image Regression Datamodule for Testing."""

    def __init__(self) -> None:
        """Initialize a new instance of Toy Image Regression Datamodule."""
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        """Return Train Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=2)

    def val_dataloader(self) -> DataLoader:
        """Return Val Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=2)

    def test_dataloader(self) -> DataLoader:
        """Return Test Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=2)
