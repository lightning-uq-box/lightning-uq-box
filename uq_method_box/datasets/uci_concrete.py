"""UCI Concrete Dataset."""

from typing import Tuple

import numpy as np
import pandas as pd

from uq_method_box.datasets.base import UCIRegressionDataset


class UCIConcrete(UCIRegressionDataset):
    """UCI Conrete Dataset."""

    data_url = "concrete/compressive/Concrete_Data.xls"

    dataset_name = "concrete"

    filename = "Concrete_Data.xls"

    def __init__(
        self,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
    ) -> None:
        """Initialize a new instance of UCIConcrete dataset.

        Args:
            root: dataset root where you want to store the 'concrete' dataset,
                a subdirectory for 'concrete' will be created automatically
            train_size: proportion of data that should be used for training
            seed: seed to randomly split data
        """
        super().__init__(root, train_size, seed, calibration_set)

    def load_data(self) -> Tuple[np.ndarray]:
        """Load the Concrete dataset."""
        data = pd.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)
