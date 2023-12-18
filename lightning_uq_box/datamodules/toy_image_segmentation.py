# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.
"""Toy image segmentation Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToySegmentationDataset


class ToySegmentationDataModule(LightningDataModule):
    """Toy segmentation datamodule."""

    def __init__(self, num_images: int=10, image_size: int=64, num_classes: int = 4, batch_size: int =10):
        """Initialize a toy image segmentation datamodule.

        Args:
            num_images: number of images in the dataset
            image_size: size of the image
            batch_size: batch size
        """
        super().__init__()
        self.num_images = num_images
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            ToySegmentationDataset(self.num_images, self.image_size, self.num_classes),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the val dataloader."""
        return DataLoader(
            ToySegmentationDataset(self.num_images, self.image_size, self.num_classes),
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            ToySegmentationDataset(self.num_images, self.image_size, self.num_classes),
            batch_size=self.batch_size,
        )
