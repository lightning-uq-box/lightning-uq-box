"""UCI Yacht Dataset."""


import numpy as np
import pandas as pd

from uq_method_box.datasets.uci import UCIRegressionDataset


class UCIYacht(UCIRegressionDataset):
    """UCI Yacht Housing Dataset."""

    data_url = "/00243/yacht_hydrodynamics.data"

    dataset_name = "yacht"

    filename = "yacht_hydrodynamics.data"

    def __init__(
        self,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
    ) -> None:
        """Initialize a new instance of UCIYacht dataset.

        Args:
            root: dataset root where you want to store the 'yacht' dataset,
                a subdirectory for 'yacht' will be created automatically
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__(root, train_size, seed, calibration_set)

    def load_data(self) -> tuple[np.ndarray]:
        """Load the Yacht dataset."""
        data = pd.read_fwf(self.datapath, header=None).values[:-1, :]
        return data[:, :-1], data[:, -1].reshape(-1, 1)
