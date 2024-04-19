# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Toy Pixelwise Regression Dataset."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyPixelWiseRegressionDataset(Dataset):
    """Toy pixel-wise regression dataset."""

    def __init__(self, num_images: int = 10, image_size: int = 64):
        """Initialize a toy pixel-wise regression dataset.

        Args:
            num_images: number of images in the dataset
            image_size: size of the image
        """
        self.num_images = num_images
        self.image_size = image_size

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.num_images

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Generate a random grayscale image and corresponding target.

        Args:
            idx: index of the sample
        """
        image = torch.randint(
            0, 1, (3, self.image_size, self.image_size), dtype=torch.float32
        )
        target = torch.rand(1, self.image_size, self.image_size, dtype=torch.float32)

        return {
            "input": image,
            "target": target,
            "index": idx,
            "aux": "random_aux_data",
        }
