"""UCI Regression Datamodule."""

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from uq_method_box.datasets import (
    UCIBoston,
    UCIConcrete,
    UCIEnergy,
    UCINaval,
    UCIRegressionDataset,
    UCIYacht,
)


class UCIRegressionDatamodule(LightningDataModule):
    """Datamodule class for all UCI Regression Datasets."""

    def __init__(
        self,
        dataset_name: str,
        root: str,
        train_size: float = 0.9,
        seed: int = 0,
        calibration_set: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Initialize a new instance of the Datamodule.

        Args:
            dataset_name: which uci dataset to load
            root: root directory where uci dataset is placed or should be downloaded
            train_size: size of the training set
            seed: seed with which to do the dataset split
            calibration_set: whether to create an additional calibration set for
                conformal prediction for example or validation
            batch_size: batch_size for dataloaders
            num_workers: number of workers dataloading process
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.root = root
        self.train_size = train_size
        self.seed = seed
        self.calibration_set = calibration_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uci_ds = self.initialize_dataset()

    def initialize_dataset(self) -> UCIRegressionDataset:
        """Initialize the desired UCI Regression Dataset.

        Args:
            config: config dictionary

        Returns:
            the initialized UCI Regression Dataset
        """
        dataset_class = {
            "boston": UCIBoston,
            "naval": UCINaval,
            "concrete": UCIConcrete,
            "yacht": UCIYacht,
            "energy": UCIEnergy,
        }

        return dataset_class[self.dataset_name](
            root=self.root,
            train_size=self.train_size,
            seed=self.seed,
            calibration_set=self.calibration_set,
        )

    def train_dataloader(self) -> DataLoader:
        """Return a dataloader for the training set."""
        return DataLoader(
            self.uci_ds.train_dataset(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a dataloader for the testing set."""
        return DataLoader(
            self.uci_ds.test_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def calibration_dataloader(self) -> DataLoader:
        """Return a calibration dataloader for conformal prediction."""
        return DataLoader(
            self.uci_ds.calibration_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
