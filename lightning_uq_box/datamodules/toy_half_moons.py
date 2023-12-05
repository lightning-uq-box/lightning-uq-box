# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Two Moons Toy Classification Datamodule."""

from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import collate_fn_tensordataset


class TwoMoonsDataModule(LightningDataModule):
    """DataModule for PyTorch Lightning that encapsulates the half-moon dataset."""

    def __init__(self, batch_size: int = 32, n_samples: int = 1000):
        """Initialize the DataModule.

        Args:
            batch_size: The batch size for the DataLoaders
            n_samples: The total number of samples in the dataset
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.setup()

    def setup(self, stage: Optional[str] = None):
        """Set up the DataModule.

        Args:
            stage: The stage ('fit' or 'test'). Defaults to None.
        """
        # Generate the half-moon dataset
        X, y = make_moons(n_samples=self.n_samples, noise=0.1)

        # Convert the numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Split the dataset into training, validation, and test sets
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25
        )

        # Create a grid of test points
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
        )
        self.test_grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).to(
            torch.float32
        )

    def train_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for the training set.

        Returns:
            The DataLoader for the training set.
        """
        train_dataset = TensorDataset(self.X_train.float(), self.y_train)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for the validation set.

        Returns:
            The DataLoader for the validation set.
        """
        val_dataset = TensorDataset(self.X_val.float(), self.y_val)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for the test set.

        Returns:
            The DataLoader for the test set.
        """
        test_dataset = TensorDataset(self.X_test.float(), self.y_test)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )
