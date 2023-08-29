"""UCI Energy Dataset."""


import numpy as np
import pandas as pd

from lightning_uq_box.datasets.uci import UCIRegressionDataset


class UCIEnergy(UCIRegressionDataset):
    """UCI Energy Dataset."""

    data_url = "00242/ENB2012_data.xlsx"
    dataset_name = "energy"
    filename = "ENB2012_data.xlsx"

    def __init__(
        self,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
    ) -> None:
        """Initialize a new instance of UCIEnergy dataset.

        Args:
            root: dataset root where you want to store the 'energy' dataset,
                a subdirectory for 'energy' will be created automatically
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__(root, train_size, seed, calibration_set)

    def load_data(self) -> tuple[np.ndarray]:
        """Load the Energy dataset."""
        data = pd.read_excel(self.datapath).values[:, :-1]
        return data[:, :-1], data[:, -1].reshape(-1, 1)
