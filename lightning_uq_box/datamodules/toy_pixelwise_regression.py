# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Toy image segmentation Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lightning_uq_box.datasets import ToyPixelWiseRegressionDataset


class ToyPixelwiseRegressionDataModule(LightningDataModule):
    """Toy segmentation datamodule."""

    def __init__(self, num_images=10, image_size=64, batch_size=10):
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

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            ToyPixelWiseRegressionDataset(self.num_images, self.image_size),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the val dataloader."""
        return DataLoader(
            ToyPixelWiseRegressionDataset(self.num_images, self.image_size),
            batch_size=self.batch_size,
        )
    
    def calib_dataloader(self) -> DataLoader:
        """Return the calib dataloader."""
        return DataLoader(
            ToyPixelWiseRegressionDataset(self.num_images, self.image_size),
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            ToyPixelWiseRegressionDataset(self.num_images, self.image_size),
            batch_size=self.batch_size,
        )