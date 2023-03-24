"""Base datasets that can be used from other datasets."""

import os
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    def __init__(
        self,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
    ) -> None:
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
        self.calibration_set = calibration_set

        self.url = self.uci_base_url + self.data_url

        self.datapath = os.path.join(self.root, self.dataset_name, self.filename)

        # verify dataset
        self.verify()

        # load data
        X, y = self.load_data()

        self.N = X.shape[0]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=self.BASE_SEED + seed,
            shuffle=True,
        )

        if calibration_set:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train,
                y_train,
                train_size=train_size,
                random_state=self.BASE_SEED + seed,
                shuffle=True,
            )

        # this is how the Wilson group does it in their paper
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X)
        self.X_train = self.input_scaler.transform(X_train)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y)
        self.y_train = self.target_scaler.transform(y_train)

        if calibration_set:
            self.X_calib = self.input_scaler.transform(X_calib)
            self.y_calib = self.target_scaler.transform(y_calib)

        self.X_test = self.input_scaler.transform(X_test)
        self.y_test = self.target_scaler.transform(y_test)

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

    def calibration_dataset(self) -> TensorDataset:
        """Calibration dataset for conformal prediction experiments.

        Returns:
            TensorDataset from calibration data.
        """
        if self.calibration_set:
            return TensorDataset(
                torch.from_numpy(self.X_calib).to(torch.float32),
                torch.from_numpy(self.y_calib).to(torch.float32),
            )
        else:
            raise ValueError

    def test_dataset(self) -> TensorDataset:
        """Create the Testing Dataset.

        Returns:
            TensorDataset from test data.
        """
        return TensorDataset(
            torch.from_numpy(self.X_test).to(torch.float32),
            torch.from_numpy(self.y_test).to(torch.float32),
        )

    # def compute_normalization_statistics(
    #     self, X_train: np.ndarray, y_train: np.ndarray
    # ) -> None:
    #     """Compute the normalization statistics.

    #     Args:
    #         X_train: training features of shape [N x num_features]
    #         y_train: training targets of shape [N x 1]
    #     """
    #     self.X_mean, self.X_std = X_train.mean(axis=0), X_train.std(axis=0)
    #     self.y_mean, self.y_std = y_train.mean(axis=0), y_train.std(axis=0)

    # def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    #     """Preprocess the training and testing data.

    #     Args:
    #         X: feature matrix of shape [N x num_features]
    #         y: targets of shape [N x 1]

    #     Returns:
    #         processed versions of data
    #     """
    #     return (X - self.X_mean) / self.X_std, (y - self.y_mean) / self.y_std

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
