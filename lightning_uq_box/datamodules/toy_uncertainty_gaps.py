"""Toy Uncertainty Gaps."""

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class ToyUncertaintyGaps(LightningDataModule):
    """Toy Uncertainty Gaps."""

    def __init__(self) -> None:
        """Initialize a new instance of DataModule."""
        super().__init__()

        def features(x):
            return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

        data = np.load(
            "/home/nils/projects/uq-method-box/lightning_uq_box/datamodules/data.npy"
        )
        x, y = data[:, 0], data[:, 1]
        y = y[:, None]
        f = features(x)

        self.X_test = torch.from_numpy(features(np.linspace(-10, 10, 100))).to(
            torch.float32
        )
        self.y_test = torch.rand(self.X_test.shape[0]).to(torch.float32) * 0.1
        self.X_test_plot = np.linspace(-10, 10, 100)

        self.X_train_plot = x

        self.X_train = torch.from_numpy(f.astype(np.float32))
        self.y_train = torch.from_numpy(y.astype(np.float32))

    def train_dataloader(self) -> DataLoader:
        """Train loader."""
        return DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=50)

    def test_dataloader(self) -> DataLoader:
        """Test loader."""
        return DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=50)
