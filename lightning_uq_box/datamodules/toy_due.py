# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Dataset from DUE repository.

Adapted from: https://github.com/y0ast/DUE/blob/main/toy_regression.ipynb

"""

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .utils import collate_fn_tensordataset


class ToyDUE(LightningDataModule):
    """Toy Dataset from DUE repository."""

    def __init__(
        self,
        n_samples: int = 500,
        noise: float = 0.2,
        batch_size: int = 200,
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        """Initialize a new Toy Data Module instance.

        Args:
            n_samples: number of samples for dataset
            noise: gaussian noise variance
            batch_size: batch size for data loaders
            test_fraction: fraction of n_points for test set
            val_fraction: fraction of n_points for validation
                set
            split_seed: random seed for data split
        """
        super().__init__()

        self.batch_size = batch_size
        # make some random sines & cosines
        np.random.seed(2)
        n_samples = int(n_samples)

        W = np.random.randn(30, 1)
        b = np.random.rand(30, 1) * 2 * np.pi

        self.X_all = np.sort(
            5 * np.sign(np.random.randn(n_samples))
            + np.random.randn(n_samples).clip(-2, 2)
        )
        self.Y_all = np.cos(W * self.X_all + b).sum(0) + noise * np.random.randn(
            n_samples
        )
        self.X_all = self.X_all.reshape(-1, 1)
        self.Y_all = self.Y_all.reshape(-1, 1)

        # split data into train and held out IID test
        X_other, self.X_test, Y_other, self.Y_test = train_test_split(
            self.X_all, self.Y_all, test_size=test_fraction, random_state=split_seed
        )

        # split train data into train and validation
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X_other,
            Y_other,
            test_size=val_fraction / (1 - test_fraction),
            random_state=split_seed,
        )

        self.X_gtext = np.linspace(-10, 10, 500)
        self.Y_gtext = np.cos(W * self.X_gtext + b).sum(0)
        self.X_gtext = self.X_gtext.reshape(-1, 1)
        self.Y_gtext = self.Y_gtext.reshape(-1, 1)

        scalers = dict(
            X=StandardScaler().fit(self.X_train), Y=StandardScaler().fit(self.Y_train)
        )
        for xy in ["X", "Y"]:
            for arr_type in ["train", "val", "test", "gtext"]:
                arr_name = f"{xy}_{arr_type}"
                setattr(
                    self,
                    arr_name,
                    self._n2t(scalers[xy].transform(getattr(self, arr_name))),
                )

    @staticmethod
    def _n2t(x):
        return torch.from_numpy(x).type(torch.float32)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.Y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        # TODO Validation data
        return DataLoader(
            TensorDataset(self.X_train, self.Y_train),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.Y_test),
            batch_size=self.X_test.shape[0],
            shuffle=False,
            collate_fn=collate_fn_tensordataset,
        )
