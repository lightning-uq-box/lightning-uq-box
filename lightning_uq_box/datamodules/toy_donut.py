# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyDonut


class ToyDonutDataModule(LightningDataModule):
    """DataModule for Toy Donut."""

    def __init__(self, batch_size: int = 64, n_samples: int = 10000):
        """Initialize the DataModule.

        Args:
            batch_size: The batch size for the DataLoader. Defaults to 64.
        """
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = ToyDonut(n_samples=n_samples)
        self.X_train, self.y_train = self.train_dataset.X, self.train_dataset.y
        self.val_dataset = ToyDonut(n_samples=int(n_samples * 0.2))
        self.X_val, self.y_val = self.val_dataset.X, self.val_dataset.y
        self.test_dataset = ToyDonut(n_samples=int(n_samples * 0.4))
        self.X_test, self.y_test = self.test_dataset.X, self.test_dataset.y

    def train_dataloader(self):
        """Create the train DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """Create the validation DataLoader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Create the test DataLoader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
