"""UCI Regression Datamodule."""

from typing import Any, Dict

from pytorch_lightning import LightningDataModule
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

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize a new instance of the Datamodule.

        Args:
            config: config dictionary
        """
        super().__init__()

        self.config = config
        self.uci_ds = self.initialize_dataset(config)

    def initialize_dataset(self, config: Dict[str, Any]) -> UCIRegressionDataset:
        """Initialize the desired UCI Regression Dataset.

        Args:
            config: config dictionary

        Returns:
            the initialized UCI Regression Dataset
        """
        dataset_name = config["ds"]["dataset_name"]

        dataset_class = {
            "boston": UCIBoston,
            "naval": UCINaval,
            "concrete": UCIConcrete,
            "yacht": UCIYacht,
            "energy": UCIEnergy,
        }

        dataset_args = {
            arg: val for arg, val in config["ds"].items() if arg not in ["dataset_name"]
        }
        dataset_args["calibration_set"] = config["model"].get("conformalize", False)

        return dataset_class[dataset_name](**dataset_args)

    def train_dataloader(self) -> DataLoader:
        """Return a dataloader for the training set."""
        return DataLoader(self.uci_ds.train_dataset(), **self.config["dataloader"])

    def test_dataloader(self) -> DataLoader:
        """Return a dataloader for the testing set."""
        return DataLoader(self.uci_ds.test_dataset(), **self.config["dataloader"])

    def calibration_dataloader(self) -> DataLoader:
        """Return a calibration dataloader for conformal prediction."""
        return DataLoader(
            self.uci_ds.calibration_dataset(), **self.config["dataloader"]
        )
