"""Base datasets that can be used from other datasets."""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchgeo.datasets.utils import download_and_extract_archive
from torchvision.datasets.utils import download_url


class UCIRegressionDataset:
    """Base Class for UCI Regression Datasets."""

    uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    data_url = "dataset"
    BASE_SEED = 0
    dataset_name = "base"
    filename = "dataset_filename"

    def __init__(self, root: str, train_size: float = 0.9, seed: int = 0) -> None:
        """Initiate a new instance of UCI Regression Dataset.

        Args:
            root: dataset root where datasets can be found under the subdirectory
                of *self.dataset_name*
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__()
        self.root = root
        self.train_size = train_size

        self.url = self.uci_base_url + self.data_url

        self.datapath = os.path.join(self.root, self.dataset_name, self.filename)

        # verify dataset
        self.verify()

        # load data
        X, y = self.load_data()

        # split data into train and test
        ind = np.arange(X.shape[0])
        np.random.seed(self.BASE_SEED + seed)
        np.random.shuffle(ind)

        n = int(X.shape[0] * self.train_size)

        X_train = X[ind[:n]]
        y_train = y[ind[:n]]

        X_test = X[ind[n:]]
        y_test = y[ind[n:]]

        # compute normalization statistics on train set
        self.compute_normalization_statistics(X_train, y_train)

        # normalize data
        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess(
            X_train, y_train, X_test, y_test
        )

        self.num_features = self.X_train.shape[-1]

    def load_data(self) -> Tuple[np.ndarray]:
        """Load the data from the file."""
        raise NotImplementedError

    def train_dataset(self) -> TensorDataset:
        """Create the Training Dataset.

        Returns:
            TensorDataset from training data.
        """
        return TensorDataset(
            torch.from_numpy(self.X_train).to(torch.float32),
            torch.from_numpy(self.y_train).to(torch.float32),
        )

    def test_dataset(self) -> TensorDataset:
        """Create the Testing Dataset.

        Returns:
            TensorDataset from test data.
        """
        return TensorDataset(
            torch.from_numpy(self.X_test).to(torch.float32),
            torch.from_numpy(self.y_test).to(torch.float32),
        )

    def compute_normalization_statistics(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Compute the normalization statistics.

        Args:
            X_train: training features of shape [N x num_features]
            y_train: training targets of shape [N x 1]
        """
        self.X_mean, self.X_std = X_train.mean(axis=0), X_train.std(axis=0)
        self.y_mean, self.y_std = y_train.mean(axis=0), y_train.std(axis=0)

    def preprocess(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """Preprocess the training and testing data.

        Args:
            X_train: training features of shape [N x num_features]
            y_train: training targets of shape [N x 1]
            X_test: testing features of shape [N* x num_features]
            y_test: testing targets of shape [N* x 1]

        Returns:
            processed versions of data
        """
        X_train = (X_train - self.X_mean) / self.X_std
        y_train = (y_train - self.y_mean) / self.y_std

        X_test = (X_test - self.X_mean) / self.X_std
        y_test = (y_test - self.y_mean) / self.y_std

        return X_train, y_train, X_test, y_test

    def verify(self) -> None:
        """Verify presence of data."""
        data_dir = os.path.join(self.root, self.dataset_name)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        data_filepath = os.path.join(data_dir, self.filename)
        if os.path.isfile(data_filepath):
            return  # dataset present

        # otherwise download data
        self.download_data()

    def download_data(self) -> None:
        """Download the dataset to root."""
        print(f"Downloading dataset {self.dataset_name} to directory {self.root}")

        is_zipped = np.any([z in self.url for z in [".gz", ".zip", ".tar"]])

        if is_zipped:
            download_and_extract_archive(
                self.url,
                os.path.dirname(self.datapath),
                os.path.dirname(os.path.dirname(self.datapath)),
            )
        else:
            download_url(self.url, os.path.dirname(self.datapath), self.filename)

        print(f"finished donwloading {self.dataset_name}")
