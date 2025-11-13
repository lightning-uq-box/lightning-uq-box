# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Image Regression Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageRegressionDataset


class ToyImageRegressionDatamodule(LightningDataModule):
    """Toy Image Regression Datamodule for Testing."""

    def __init__(self, num_samples: int = 4, batch_size: int = 10) -> None:
        """Initialize a new instance of Toy Image Regression Datamodule.

        Args:
            batch_size: batch size
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        """Return Train Dataloader."""
        return DataLoader(
            ToyImageRegressionDataset(self.num_samples), batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Return Val Dataloader."""
        return DataLoader(
            ToyImageRegressionDataset(self.num_samples), batch_size=self.batch_size
        )

    def calib_dataloader(self) -> DataLoader:
        """Return Calib Dataloader."""
        return DataLoader(
            ToyImageRegressionDataset(self.num_samples), batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Return Test Dataloader."""
        return DataLoader(
            ToyImageRegressionDataset(self.num_samples), batch_size=self.batch_size
        )
