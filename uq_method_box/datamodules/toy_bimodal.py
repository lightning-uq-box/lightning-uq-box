"""Toy Bimodal Dataset."""

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class ToyBimodalDatamodule(LightningDataModule):
    """Bimodal Toy Regression Datamodule.

    Dataset is the same for train, val, test.
    """

    def __init__(self, N: int = 500, batch_size: int = 100) -> None:
        """Instantiate a Bimodal Regression Toy Dataset.

        Args:
            N: number of datapoints
            batch_size: batch size for dataloader
        """
        super().__init__()
        self.N = N
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

        self.X = torch.from_numpy(X).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(TensorDataset(self.X, self.y), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(TensorDataset(self.X, self.y), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(TensorDataset(self.X, self.y), batch_size=self.batch_size)
