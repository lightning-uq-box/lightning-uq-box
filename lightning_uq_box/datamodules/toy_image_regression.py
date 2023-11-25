# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Toy Image Regression Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageRegressionDataset


class ToyImageRegressionDatamodule(LightningDataModule):
    """Toy Image Regression Datamodule for Testing."""

    def __init__(self, batch_size: int = 10) -> None:
        """Initialize a new instance of Toy Image Regression Datamodule.

        Args:
            batch_size: batch size
        """
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """Return Train Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return Val Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return Test Dataloader."""
        return DataLoader(ToyImageRegressionDataset(), batch_size=self.batch_size)
