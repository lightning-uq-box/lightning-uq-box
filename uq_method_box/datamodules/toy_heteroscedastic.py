"""Datamodule for Toy Heteroscedastic Data."""

from typing import Callable, Union

import numpy as np
import torch
from lightning import LightningDataModule
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


def polynomial_f(x):
    """Polynomial function used to generate one-dimensional data."""
    return np.array(5 * x + 5 * x**4 - 9 * x**2)


def linear_f(x):
    """Linear function to generate one-dimensional data."""
    return x


def polynomial_f2(x):
    """Polynomial function."""
    w = np.array([-0.6667, -0.6012, -1.0172, -0.7687, 2.4680, -0.1678])
    fx = 0
    for i in range(len(w)):
        fx += w[i] * (x**i)
    fx *= np.sin(np.pi * x)
    fx *= np.exp(-0.5 * (x**2)) / np.sqrt(2 * np.pi)
    return fx


class ToyHeteroscedasticDatamodule(LightningDataModule):
    """Implement Toy Dataset with heteroscedastic noise."""

    def __init__(
        self,
        x_min: Union[int, float] = -4,
        x_max: Union[int, float] = 4,
        n_train: int = 200,
        n_true: int = 200,
        sigma: float = 0.3,
        batch_size: int = 200,
        generate_y: Callable = polynomial_f2,
    ) -> None:
        """Define a heteroscedastic toy regression dataset.

        Taken from (https://mapie.readthedocs.io/en/latest/examples_regression/
        1-quickstart/plot_heteroscedastic_1d_data.html#sphx-glr-examples-
        regression-1-quickstart-plot-heteroscedastic-1d-data-py)

        Args:
            n_train : Number of training samples, by default  200.
            n_true: Number of test samples, by default 1000.
            sigma: Standard deviation of noise, by default 0.1
            batch_size: batch size for data loader
            generate_y: function that should generate data over input line
        """
        super().__init__()
        np.random.seed(1)
        q95 = stats.norm.ppf(0.95)
        X_train = np.linspace(x_min, x_max, n_train)
        X_test = np.linspace(x_min + 0.1 * x_min, x_max + x_max * 0.3, n_true)
        y_train = generate_y(X_train) + np.random.normal(
            0, sigma, n_train
        ) * np.linspace(0.1, 1, n_train)
        y_true = generate_y(X_test)

        # "True" noise
        y_true_sigma = q95 * sigma * X_test  # noqa: F841

        # train loader
        gap_start = x_min + 0.4 * (x_max - x_min)
        gap_end = x_min + 0.6 * (x_max - x_min)

        test_idx = ((X_train > gap_start) & (X_train < gap_end)).squeeze()

        X_train = torch.from_numpy(X_train).unsqueeze(-1).type(torch.float32)
        self.X_train = X_train[~test_idx]
        y_train = torch.from_numpy(y_train).unsqueeze(-1).type(torch.float32)
        self.y_train = y_train[~test_idx]

        self.X_test = torch.from_numpy(X_test).unsqueeze(-1).type(torch.float32)
        self.y_test = torch.from_numpy(y_true).unsqueeze(-1).type(torch.float32)

        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        # TODO Validation data
        return DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size
        )
