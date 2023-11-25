"""Datamodule for Toy Sinusoidal example."""

from typing import Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class ToySineDatamodule(LightningDataModule):
    """Implement a Datamodule for Toy Sinusoidal Example."""

    def __init__(
        self,
        n_data: int = 500,
        sigma_noise_1: float = 0.1,
        sigma_noise_2: float = 0.4,
        x_min: Union[int, float] = -2,
        x_max: Union[int, float] = 15,
        batch_size: int = 500,
    ) -> None:
        """Define a sinosoidal toy regression dataset.

        Args:
            n_data: number of data points
            sigma_noise_1: injected sigma noise around the left
                half of the input interval
            sigma_noise_2: injected sigma noise around the right
                half of the input interval
            x_min: minimum value of the input interval
            x_max: maximum value of the input interval
            batch_size: batch size for dataloaders
        """
        super().__init__()

        X_train = (torch.linspace(x_min, x_max, n_data)).unsqueeze(-1)

        gap_start = x_min + 0.4 * (x_max - x_min)
        gap_end = x_min + 0.6 * (x_max - x_min)

        # take out validation set as gap  also for calibration of conformal prediction
        test_idx = ((X_train > gap_start) & (X_train < gap_end)).squeeze()

        noise_1 = torch.randn_like(X_train) * sigma_noise_1
        noise_1[X_train > gap_start] = 0  # only add noise to the left
        noise_2 = torch.randn_like(X_train) * sigma_noise_2
        noise_2[X_train < gap_end] = 0  # only add noise to the right

        # create simple sinusoid data set and
        # add gaussian noise with different variances
        label_noise = noise_1 + noise_2
        y_train = torch.sin(X_train) + label_noise

        # update X_train
        self.X_train = X_train[~test_idx, :]
        self.y_train = y_train[~test_idx, :]

        # test over the whole line
        self.X_test = torch.linspace(
            X_train.min() + X_train.min() * 0.1,
            X_train.max() + X_train.max() * 0.1,
            n_data,
        ).unsqueeze(-1)
        self.y_test = torch.sin(self.X_test)

        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        # TODO Validation data
        return DataLoader(
            TensorDataset(self.X_train, self.y_train), batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.y_test), batch_size=self.batch_size
        )
