# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Datamodule for Toy Multivariate Regression Data."""

import torch
from lightning import LightningDataModule
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from lightning_uq_box.datamodules.utils import collate_fn_tensordataset


class ToyMultiRegressionDataModule(LightningDataModule):
    """Implement Toy Multivariate Regression DataModule."""

    def __init__(
        self,
        n_datapoints: int = 200,
        n_features: int = 4,
        n_targets: int = 3,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """Initialize the datamodule.

        Args:
            n_datapoints: Number of datapoints to generate. Data will be split into train, validation and test sets
            n_features: Number of features of the input data
            n_targets: number of regression targets
            batch_size: Batch size for the dataloaders
            num_workers: Number of workers for the dataloaders
        """
        super().__init__()

        self.n_datapoints = n_datapoints
        self.n_features = n_features
        self.n_targets = n_targets

        X_all, y_all = make_regression(
            n_samples=self.n_datapoints,
            n_targets=self.n_targets,
            n_features=self.n_features,
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_all, y_all, test_size=0.2
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        # convert to tensors
        self.X_train = self._n2t(self.X_train)
        self.X_val = self._n2t(self.X_val)
        self.X_test = self._n2t(self.X_test)

        self.y_train = self._n2t(self.y_train)
        self.y_val = self._n2t(self.y_val)
        self.y_test = self._n2t(self.y_test)

        self.train = TensorDataset(self.X_train, self.y_train)
        self.val = TensorDataset(self.X_val, self.y_val)
        self.test = TensorDataset(self.X_test, self.y_test)

    @staticmethod
    def _n2t(x):
        return torch.from_numpy(x).type(torch.float32)

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn_tensordataset,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn_tensordataset,
        )
