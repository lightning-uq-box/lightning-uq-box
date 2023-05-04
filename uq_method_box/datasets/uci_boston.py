"""UCI Boston Dataset."""


import numpy as np
import pandas as pd

from uq_method_box.datasets.uci import UCIRegressionDataset


class UCIBoston(UCIRegressionDataset):
    """UCI Boston Housing Dataset."""

    data_url = "housing/housing.data"

    dataset_name = "boston"

    filename = "housing.data"

    def __init__(
        self,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
    ) -> None:
        """Initialize a new instance of UCIBoston dataset.

        Args:
            root: dataset root where you want to store the 'boston' dataset,
                a subdirectory for 'boston' will be created automatically
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__(root, train_size, seed, calibration_set)

    def load_data(self) -> tuple[np.ndarray]:
        """Load the Boston dataset."""
        data = pd.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)
