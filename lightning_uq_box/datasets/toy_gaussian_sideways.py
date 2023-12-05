# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ToyGaussianSideWays(Dataset):
    """Gaussian Sideways for Regression."""

    def __init__(
        self,
        radius: float = 1.0,
        n_samples: int = 1000,
        noise: float = 0.1,
        random_state: bool = None,
    ):
        """Initialize a new instance of the dataset.

        Args:
            radius: The radius of the circle. Defaults to 1.0.
            n_samples: The total number of samples in the dataset. Defaults to 1000.
            noise: Standard deviation of Gaussian noise added to the data. Defaults to 0.1.
            random_state: Determines random number generation for dataset creation. Defaults to None.
        """
        self.radius = radius
        self.n_samples = n_samples
        self.noise = noise
        np.random.seed(random_state)

        # Generate data
        self.X, self.y = self._generate_data()

    def __len__(self):
        """Get the length of the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        return {"input": self.X[idx], "target": self.y[idx]}

    def _generate_data(self):
        """Generate the dataset.

        Returns:
            The generated dataset.
        """
        # Generate uniform random angles
        angles = 2 * np.pi * np.random.rand(self.n_samples)

        # Generate data points on the circle
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)

        # Add Gaussian noise to the data
        x += self.noise * np.random.randn(self.n_samples)
        y += self.noise * np.random.randn(self.n_samples)

        # Normalize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(x.reshape(-1, 1))

        # Convert to tensors
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(angles).float().unsqueeze(-1)

        return X, y
