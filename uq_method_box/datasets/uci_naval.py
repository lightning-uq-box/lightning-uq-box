"""UCI Naval Dataset."""

from typing import Tuple

import numpy as np
import pandas as pd

from uq_method_box.datasets.base import UCIRegressionDataset


class UCINaval(UCIRegressionDataset):
    """UCI Naval Dataset."""

    data_url = "00316/UCI%20CBM%20Dataset.zip"

    dataset_name = "naval"

    filename = "UCI CBM Dataset/data.txt"

    def __init__(self, root: str, train_size: float = 0.9, seed: int = 0) -> None:
        """Initialize a new instance of UCINaval dataset.

        Args:
            root: dataset root where you want to store the 'naval' dataset,
                a subdirectory for 'naval' will be created automatically
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__(root, train_size, seed)

    def load_data(self) -> Tuple[np.ndarray]:
        """Load the Naval dataset."""
        data = pd.read_fwf(self.datapath, header=None).values
        # NB this is the first output
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)

        # dims 8 and 11 have std=0:
        X = np.delete(X, [8, 11], axis=1)
        return X, Y
