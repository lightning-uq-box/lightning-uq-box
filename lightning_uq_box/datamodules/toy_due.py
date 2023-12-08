# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Toy Dataset from DUE repository.

Adapted from: https://github.com/y0ast/DUE/blob/main/toy_regression.ipynb

"""

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from .utils import collate_fn_tensordataset


class ToyDUE(LightningDataModule):
    """Toy Dataset from DUE repository."""

    def __init__(
        self, n_samples: int = 500, noise: float = 0.2, batch_size: int = 200
    ) -> None:
        """Initialize a new Toy Data Module instance.

        Args:
            n_samples: number of samples for dataset
            noise: gaussian noise variance
            batch_size: batch size for data loaders
        """
        super().__init__()

        self.batch_size = batch_size
        # make some random sines & cosines
        np.random.seed(2)
        n_samples = int(n_samples)

        W = np.random.randn(30, 1)
        b = np.random.rand(30, 1) * 2 * np.pi

        x = np.sort(
            5 * np.sign(np.random.randn(n_samples))
            + np.random.randn(n_samples).clip(-2, 2)
        )
        y = np.cos(W * x + b).sum(0) + noise * np.random.randn(n_samples)

        x_test = np.linspace(-10, 10, 500)
        # x_test = np.sort(
        #     6.5 * np.sign(np.random.randn(n_samples))
        #     + np.random.randn(n_samples)
        # )
        y_test = np.cos(W * x_test + b).sum(0)  # + noise * np.random.randn(n_samples)

        self.X_train = torch.from_numpy(x).unsqueeze(-1).to(torch.float32)
        self.y_train = torch.from_numpy(y).unsqueeze(-1).to(torch.float32)

        self.X_test = torch.from_numpy(x_test).unsqueeze(-1).to(torch.float32)
        self.y_test = torch.from_numpy(y_test).unsqueeze(-1).to(torch.float32)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        # TODO Validation data
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.X_test.shape[0],
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )
