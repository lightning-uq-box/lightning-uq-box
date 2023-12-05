# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyDonut(Dataset):
    """Toy Donut for Regression."""

    def __init__(
        self,
        inner_radius: float = 8.0,
        outer_radius: float = 10.0,
        n_samples: int = 1000,
        noise=0.1,
    ):
        """Initialize a new instance of the dataset.

        Args:
            inner_radius: The inner radius of the donut.
            outer_radius: The outer radius of the donut.
            n_samples: The total number of samples in the dataset.
            noise: The amount of noise to add to the data.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.n_samples = n_samples
        self.noise = noise

        # Generate uniform random angles
        self.theta = 2 * np.pi * torch.rand(n_samples)

        # Generate uniform random radii within the donut
        self.radii = (
            torch.rand(n_samples) * (outer_radius - inner_radius) + inner_radius
        )

        # Generate the x and y values
        self.X = (self.radii + self.noise * torch.randn(n_samples)).float() * torch.cos(
            self.theta
        )
        self.X = self.X.unsqueeze(-1)
        self.y = (self.radii + self.noise * torch.randn(n_samples)).float() * torch.sin(
            self.theta
        )
        self.y = self.y.unsqueeze(-1)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.n_samples

    def __getitem__(self, idx) -> dict[str, Tensor]:
        """Return a sample from the dataset.

        Args:
            idx: The index of the sample to return.

        Returns:
            A dictionary containing the input and target values.
        """
        return {"input": self.X[idx], "target": self.y[idx]}
