# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Datamodule for Toy Heteroscedastic Data."""

from collections.abc import Callable

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def noisy_sine(x, noise: bool = True):
    """(Noisy) sine function."""
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
        n_points: int = 250,
        batch_size: int = 100,
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        calib_fraction: float = 0.4,
        generate_y: Callable = noisy_sine,
        noise_seed: int = 42,
        split_seed: int = 42,
        invert: bool = False,
    ) -> None:
        """Define a heteroscedastic toy regression dataset.

        Inspired by (https://mapie.readthedocs.io/en/latest/examples_regression/
        1-quickstart/plot_heteroscedastic_1d_data.html#sphx-glr-examples-
        regression-1-quickstart-plot-heteroscedastic-1d-data-py)

        Split `n_points` data points into train, validation and test set. We
        provide the following arrays: X_<type>, Y_<type>, where <type> is one
        of:

        * all: train + val + test
        * train: all - test - val
        * test: see `test_fraction`
        * val: validation, see `val_fraction`
        * calib: calibration, see `calib_fraction`
        * gtext: noise-free ground truth on extended x-axis

        X and Y have shape (n_points_{type}, 1).

        Args:
            x_min: Minimum value of x range
            x_max: Maximum value of x range
            n_points : Number of train + test + validation points
            batch_size: batch size for data loader
            test_fraction: fraction of n_points for test set
            val_fraction: fraction of n_points for validation
                set
            calib_fraction: fraction of n_points for calibration set
                will be split from validation set and necessary for
                conformal prediction
            generate_y: ground truth function with noise option, must have
                signature f(x, noise: bool)
            noise_seed: random seed for x points positions and y noise
            split_seed: random seed for train/test/val split
            invert: whether to model the inverse problem, swaps X and Y
                in the DataLoader and variables
        """
        super().__init__()

        self.invert = invert

        # TODO: use rng=np.random.default_rng(seed) and pass to noisY_sine()
        np.random.seed(noise_seed)

        self.batch_size = batch_size

        x = np.empty(n_points)
        y = np.empty(n_points)
        for k in range(n_points):
            rnd = np.random.rand()
            if rnd < 1 / 3.0:
                x[k] = np.random.normal(loc=x_min, scale=2.0 / 5.0)
            else:
                if rnd < 2.0 / 3.0:
                    x[k] = np.random.normal(loc=0.0, scale=0.9)
                else:
                    x[k] = np.random.normal(loc=x_max, scale=2.0 / 5.0)

            y[k] = generate_y(x[k])

        # full dataset
        self.X_all = x[:, None]
        self.Y_all = y[:, None]

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

        # split validation data into validation and calibration (for conformal)
        self.X_val, self.X_calib, self.Y_val, self.Y_calib = train_test_split(
            self.X_val, self.Y_val, test_size=calib_fraction, random_state=split_seed
        )

        # Ground truth for plotting (x,y) and prediction (x)
        xmin, xmax = self.X_all.min(), self.X_all.max()
        span = xmax - xmin
        self.X_gtext = np.linspace(
            xmin - span * 0.1, xmax + span * 0.1, int(n_points * 1.5)
        )[:, None]
        self.Y_gtext = generate_y(self.X_gtext, noise=False)

        # Fit scalers on train data
        scalers = dict(
            X=StandardScaler().fit(self.X_train), Y=StandardScaler().fit(self.Y_train)
        )

        # Apply scaling to all splits, convert to torch tensors
        for xy in ["X", "Y"]:
            for arr_type in ["train", "test", "val", "gtext", "calib", "all"]:
                arr_name = f"{xy}_{arr_type}"
                setattr(
                    self,
                    arr_name,
                    self._n2t(scalers[xy].transform(getattr(self, arr_name))),
                )

        if self.invert:
            for arr_type in ["train", "test", "val", "calib", "all"]:
                X_arr_name = f"X_{arr_type}"
                Y_arr_name = f"Y_{arr_type}"
                X_data = getattr(self, X_arr_name)
                Y_data = getattr(self, Y_arr_name)
                setattr(self, X_arr_name, Y_data)
                setattr(self, Y_arr_name, X_data)

            # handle the extended line separately
            self.X_gtext = self._n2t(
                np.linspace(
                    self.Y_all.min() - span * 0.1,
                    self.Y_all.max() + span * 0.1,
                    int(n_points * 1.5),
                )[:, None]
            )
            # self.Y_gtext = self._n2t(generate_y(self.X_gtext, noise=False))
            # self.X_gtext = self._n2t(scalers["Y"].transform(self.X_gtext))

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
        return DataLoader(
            TensorDataset(self.X_val, self.Y_val),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def calib_dataloader(self) -> DataLoader:
        """Return calibration dataloader."""
        return DataLoader(
            TensorDataset(self.X_calib, self.Y_calib),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.Y_test),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def gt_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_gtext, self.Y_gtext),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
