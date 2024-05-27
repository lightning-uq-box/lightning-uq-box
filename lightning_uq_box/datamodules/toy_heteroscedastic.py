# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Datamodule for Toy Heteroscedastic Data."""

from collections.abc import Callable

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
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


def oscillating(x, noise=True):
    """Oscillating function."""
    out = 7 * np.sin(x)
    if noise:
        out += 3 * np.abs(np.cos(x / 2)) * np.random.randn()
    return out


class ToyHeteroscedasticDatamodule(LightningDataModule):
    """Implement Toy Dataset with heteroscedastic noise."""

    def __init__(
        self,
        x_min: int | float = -4,
        x_max: int | float = 4,
        n_datapoints: int = 200,
        n_ground_truth: int = 200,
        batch_size: int = 200,
        generate_y: Callable = oscillating,
    ) -> None:
        """Define a heteroscedastic toy regression dataset.

        Taken from (https://mapie.readthedocs.io/en/latest/examples_regression/
        1-quickstart/plot_heteroscedastic_1d_data.html#sphx-glr-examples-
        regression-1-quickstart-plot-heteroscedastic-1d-data-py)

        Args:
            x_min: Minimum value of x range
            x_max: Maximum value of x range
            n_datapoints : Number of datapoints that form the overall available dataset.
                10% is kept as a separate test set, of the remaining 90%,
                80% will be used for training, 12% for validation and 8% for calibration
                (necessary for conformal prediction)
            n_ground_truth: Number of noise free "ground truth" samples
            batch_size: batch size for data loader
            generate_y: function that should generate data over input line
        """
        super().__init__()
        np.random.seed(1)
        X = np.zeros(n_datapoints)
        Y = np.zeros(n_datapoints)
        for k in range(n_datapoints):
            rnd = np.random.rand()
            if rnd < 1 / 3.0:
                X[k] = np.random.normal(loc=x_min, scale=2.0 / 5.0)
            else:
                if rnd < 2.0 / 3.0:
                    X[k] = np.random.normal(loc=0.0, scale=0.9)
                else:
                    X[k] = np.random.normal(loc=x_max, scale=2.0 / 5.0)

            Y[k] = generate_y(X[k])

        # Split the training data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )
        # Separation into training and separate test set
        self.X_train, self.X_test, self.y_train, self.test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        # Separate the training data into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42
        )
        # Conformal prediction expects a calibration set separate from validation set
        self.X_val, self.X_calib, self.y_val, self.y_calib = train_test_split(
            self.X_val, self.y_val, test_size=0.4, random_state=42
        )

        # compute normlization statistics on train split
        mean_X = self.X_train.mean()
        std_X = self.X_train.std()
        mean_Y = self.y_train.mean()
        std_Y = self.y_train.std()

        def normalize_and_convert(data, mean, std):
            return ((torch.from_numpy(data).unsqueeze(-1) - mean) / std).type(
                torch.float32
            )

        # Normalize the data
        self.X_train = normalize_and_convert(self.X_train, mean_X, std_X)
        self.y_train = normalize_and_convert(self.y_train, mean_Y, std_Y)
        self.X_val = normalize_and_convert(self.X_val, mean_X, std_X)
        self.y_val = normalize_and_convert(self.y_val, mean_Y, std_Y)
        self.X_calib = normalize_and_convert(self.X_calib, mean_X, std_X)
        self.y_calib = normalize_and_convert(self.y_calib, mean_Y, std_Y)
        self.X_test = normalize_and_convert(self.X_test, mean_X, std_X)
        self.y_test = normalize_and_convert(self.test, mean_Y, std_Y)

        # separate extended test data
        X_gt = np.linspace(X.min() * 1.2, X.max() * 1.2, n_ground_truth)
        Y_gt = X_gt * 0.0
        for k in range(len(X_gt)):
            Y_gt[k] = generate_y(X_gt[k], noise=False)

        X_gt = (X_gt - mean_X) / std_X
        Y_gt = (Y_gt - mean_Y) / std_Y

        self.X_gt = normalize_and_convert(X_gt, mean_X, std_X)
        self.Y_gt = normalize_and_convert(Y_gt, mean_Y, std_Y)

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
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def calibration_dataloader(self) -> DataLoader:
        """Return calibration dataloader."""
        return DataLoader(
            TensorDataset(self.X_calib, self.y_calib),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_gt, self.Y_gt),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def gt_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.Y_test),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
