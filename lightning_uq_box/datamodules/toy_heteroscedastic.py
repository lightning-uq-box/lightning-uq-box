# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Datamodule for Toy Heteroscedastic Data."""

from typing import Callable, Union

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from .utils import collate_fn_tensordataset


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


def polynomial_f3(x, noise=True):
    """Polynomial function."""
    out = 7 * np.sin(x)
    if noise:
        out += 3 * np.abs(np.cos(x / 2)) * np.random.randn()
    return out


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
        generate_y: Callable = polynomial_f3,
    ) -> None:
        """Define a heteroscedastic toy regression dataset.

        Taken from (https://mapie.readthedocs.io/en/latest/examples_regression/
        1-quickstart/plot_heteroscedastic_1d_data.html#sphx-glr-examples-
        regression-1-quickstart-plot-heteroscedastic-1d-data-py)

        Args:
            x_min: Minimum value of x range
            x_max: Maximum value of x range
            n_train : Number of training samples, by default  200.
            n_true: Number of test samples, by default 1000.
            sigma: Standard deviation of noise, by default 0.1
            batch_size: batch size for data loader
            generate_y: function that should generate data over input line
        """
        super().__init__()
        np.random.seed(1)
        X = np.zeros(n_train)
        Y = np.zeros(n_train)
        for k in range(n_train):
            rnd = np.random.rand()
            if rnd < 1 / 3.0:
                X[k] = np.random.normal(loc=-4, scale=2.0 / 5.0)
            else:
                if rnd < 2.0 / 3.0:
                    X[k] = np.random.normal(loc=0.0, scale=0.9)
                else:
                    X[k] = np.random.normal(loc=4.0, scale=2.0 / 5.0)

            Y[k] = generate_y(X[k])

        mean_X = np.mean(X)
        std_X = np.std(X)
        mean_Y = np.mean(Y)
        std_Y = np.std(Y)
        X_n = (X - mean_X) / std_X
        Y_n = (Y - mean_Y) / std_Y

        self.X_train = torch.from_numpy(X_n).unsqueeze(-1).type(torch.float32)
        self.y_train = torch.from_numpy(Y_n).unsqueeze(-1).type(torch.float32)
        X_test = np.linspace(X.min(), X.max(), n_true)
        Y_test = X_test * 0.0
        for k in range(len(X_test)):
            Y_test[k] = generate_y(X_test[k], noise=False)

        X_test = (X_test - mean_X) / std_X
        Y_test = (Y_test - mean_Y) / std_Y
        self.X_test = torch.from_numpy(X_test).unsqueeze(-1).type(torch.float32)
        self.y_test = torch.from_numpy(Y_test).unsqueeze(-1).type(torch.float32)

        self.batch_size = batch_size

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
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
