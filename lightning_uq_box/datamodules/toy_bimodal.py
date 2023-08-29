"""Toy Bimodal Dataset."""

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class ToyBimodalDatamodule(LightningDataModule):
    """Bimodal Toy Regression Datamodule.

    Dataset is the same for train, val, test.
    """

    def __init__(
        self, n_train: int = 750, n_true: int = 200, batch_size: int = 200
    ) -> None:
        """Instantiate a Bimodal Regression Toy Dataset.
        Args:
            n_train: number of datapoints
            batch_size: batch size for dataloader
        """
        super().__init__()
        N = n_train
        self.batch_size = batch_size

        X = np.zeros((N, 1))
        y = np.zeros((N, 1))

        def f(x):
            z = np.random.randint(0, 2)
            return z * 10 * np.cos(x) + (1 - z) * 10 * np.sin(x) + np.random.randn()

        for k in range(N):
            x = np.random.exponential(0.5) - 0.5
            while x < -0.5 or x > 2:
                x = np.random.exponential(0.5) - 0.5
            X[k] = x
            y[k] = f(X[k])

        X_true = np.linspace(-0.5, 2, n_true).reshape(-1, 1)
        y_true = np.zeros((n_true, 1))
        for k in range(n_true):
            y_true[k] = f(X_true[k])

        mean_X = np.mean(X)
        std_X = np.std(X)
        mean_Y = np.mean(y)
        std_Y = np.std(y)
        X_n = (X - mean_X) / std_X
        Y_n = (y - mean_Y) / std_Y
        self.X_train = torch.from_numpy(X_n).type(torch.float32)
        self.y_train = torch.from_numpy(Y_n).type(torch.float32)
        X_test = (X_true - mean_X) / std_X
        Y_test = (y_true - mean_Y) / std_Y
        self.X_test = torch.from_numpy(X_test).type(torch.float32)
        self.y_test = torch.from_numpy(Y_test).type(torch.float32)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size
        )
