"""Toy Image Regression Dataset."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyImageRegressionDataset(Dataset):
    """Toy Image Regression Dataset."""

    def __init__(self) -> None:
        """Initialize a new instance of Toy Image Regression Dataset."""
        super().__init__()

        self.num_samples = 6
        self.images = [torch.ones(3, 64, 64) * val for val in range(self.num_samples)]
        self.targets = torch.arange(0, self.num_samples).to(torch.float32)

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> Tensor:
        """Retrieve single sample from the dataset.

        Args:
            index: index value to index dataset
        """
        return {
            "inputs": self.images[index],
            "targets": self.targets[index].unsqueeze(-1),
        }
