# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Image Regression Dataset."""

from typing import Any

import torch
from torch.utils.data import Dataset


class ToyImageRegressionDataset(Dataset):
    """Toy Image Regression Dataset."""

    def __init__(self) -> None:
        """Initialize a new instance of Toy Image Regression Dataset."""
        super().__init__()

        self.num_samples = 10
        self.images = [torch.randn(3, 64, 64) for val in range(self.num_samples)]
        self.targets = torch.arange(0, self.num_samples).to(torch.float32)

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieve single sample from the dataset.

        Args:
            index: index value to index dataset
        """
        return {
            "input": self.images[index],
            "target": self.targets[index].unsqueeze(-1),
            "index": index,
            "aux": "random_aux_data",
        }
