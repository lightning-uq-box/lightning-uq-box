# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Toy image segmentation dataset."""

from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToySegmentationDataset(Dataset):
    """Toy image segmentation dataset."""

    def __init__(
        self, num_images: int = 10, image_size: int = 64, num_classes: int = 4
    ):
        """Initialize a toy image segmentation dataset.

        Args:
            num_images: number of images in the dataset
            image_size: size of the image
            num_classes: number of classes in the dataset
        """
        self.num_images = num_images
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.num_images

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Generate a random grayscale image and corresponding mask."""
        # Generate a random grayscale image and corresponding mask
        image = torch.randint(
            0, 1, (3, self.image_size, self.image_size), dtype=torch.float32
        )
        mask = torch.randint(
            0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long
        )
        return {"input": image, "target": mask, "index": idx, "aux": "random_aux_data"}
