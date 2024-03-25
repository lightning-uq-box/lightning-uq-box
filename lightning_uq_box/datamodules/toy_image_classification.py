# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Image Classification Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyImageClassificationDataset


class ToyImageClassificationDatamodule(LightningDataModule):
    """Toy Image Classification Datamodule for Testing."""

    def __init__(self, **kwargs) -> None:
        """Initialize a new instance of Toy Image Classification Datamodule."""
        super().__init__()
        self.kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        """Return Train Dataloader."""
        return DataLoader(ToyImageClassificationDataset(**self.kwargs), batch_size=10)

    def val_dataloader(self) -> DataLoader:
        """Return Val Dataloader."""
        return DataLoader(ToyImageClassificationDataset(**self.kwargs), batch_size=10)

    def test_dataloader(self) -> DataLoader:
        """Return Test Dataloader."""
        return DataLoader(ToyImageClassificationDataset(**self.kwargs), batch_size=10)
