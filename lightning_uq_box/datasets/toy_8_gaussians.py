# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""8 Gaussians Toy Dataset."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset


class Toy8GaussiansDataset(Dataset):
    """8 Gaussians Toy Dataset."""

    def __init__(
        self,
        n_samples: int = 1000,
        radius: float = 3.0,
        std_dev: float = 0.1,
        seed: int = 0,
    ):
        """Initialize a new instance of the dataset.

        Args:
            n_samples: The total number of samples in the dataset
            radius: The radius of the circle in which the Gaussians are placed around
            std_dev: The standard deviation of the Gaussians. Defaults to 0.2
            seed: The seed for the random number generator
        """
        self.seed = seed
        self.n_samples = n_samples
        self.radius = radius
        self.points_per_gaussian = int(n_samples / 8)
        self.std_dev = std_dev
        self.data = self._generate_data()

    def __len__(self):
        """Get the length of the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx) -> dict[str, Tensor]:
        """Get a single sample from the dataset."""
        return {"input": self.X[idx], "target": self.y[idx]}

    def _generate_data(self) -> list[Tensor]:
        """Generate the dataset.

        Returns:
            The generated dataset.
        """
        np.random.seed(self.seed)
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(self.radius * x, self.radius * y) for x, y in centers]
        dataset_list = []
        for _ in range(self.points_per_gaussian):
            for i in range(8):
                point = np.random.randn(2) * self.std_dev
                center = centers[i]
                point[0] += center[0]
                point[1] += center[1]
                dataset_list.append(point)

        # Convert the list of numpy arrays to a 2D numpy array
        dataset = np.array(dataset_list)

        # Normalize the data
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)

        # Convert the 2D numpy array back to a list of tensors
        self.X = torch.tensor(
            [point[0] for point in dataset], dtype=torch.float32
        ).unsqueeze(-1)
        self.y = torch.tensor(
            [point[1] for point in dataset], dtype=torch.float32
        ).unsqueeze(-1)

        # Convert the 2D numpy array back to a list of tensors
        return [torch.tensor(point, dtype=torch.float32) for point in dataset]
