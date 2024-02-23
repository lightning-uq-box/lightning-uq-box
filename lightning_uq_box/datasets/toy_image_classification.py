# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Toy Image Classification Dataset."""

import torch
from torch.utils.data import Dataset


class ToyImageClassificationDataset(Dataset):
    """Toy Image Classification Dataset."""

    def __init__(self, num_classes: int = 4, num_samples: int = 10) -> None:
        """Initialize a new instance of Toy Image Classification Dataset."""
        super().__init__()

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.images = [torch.randn(3, 64, 64) for val in range(self.num_samples)]
        self.targets = torch.randint(0, self.num_classes, (self.num_samples,))

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        """Retrieve single sample from the dataset.

        Args:
            index: index value to index dataset
        """
        return {
            "input": self.images[index],
            "target": self.targets[index],
            "index": index,
            "aux": "random_aux_data",
        }
