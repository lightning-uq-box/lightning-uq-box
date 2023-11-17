"""Toy Image Classification Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageClassificationDataset


class ToyImageClassificationDatamodule(LightningDataModule):
    """Toy Image Classification Datamodule for Testing."""

    def __init__(self) -> None:
        """Initialize a new instance of Toy Image Classification Datamodule."""
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        """Return Train Dataloader."""
        return DataLoader(ToyImageClassificationDataset(), batch_size=2)

    def val_dataloader(self) -> DataLoader:
        """Return Val Dataloader."""
        return DataLoader(ToyImageClassificationDataset(), batch_size=2)

    def test_dataloader(self) -> DataLoader:
        """Return Test Dataloader."""
        return DataLoader(ToyImageClassificationDataset(), batch_size=2)
