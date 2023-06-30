"""USA Vars datamodule adaption for OOD experiments."""

from typing import Any, Callable, Dict

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule, USAVarsDataModule, USAVarsFeatureExtractedDataModule
from torchgeo.transforms import AugmentationSequential

from uq_method_box.datasets import USAVarsOOD


class USAVarsFeatureExtractedDataModuleOur(USAVarsFeatureExtractedDataModule):
    """USAVarsFeatureExtracted Data Module."""
    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Version we use for now."""
        super().__init__(batch_size, num_workers, **kwargs)


    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """

        return {
            "inputs": batch["image"].float(),
            "targets": (batch["labels"].float() - 23.881872) / 31.52334,
        }

class USAVarsDataModuleOur(USAVarsDataModule):
    """USA Vars Datamodule."""

    # target_mean: array([23.881872], dtype=float32)
    # target_std: array([31.52334], dtype=float32)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Version we use for now."""
        super().__init__(batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=torch.Tensor([0.4101762, 0.4342503, 0.3484594, 0.5473533]),
                std=torch.Tensor([0.17361328, 0.14048962, 0.12148701, 0.16887303]),
            ),
            K.Resize(224),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image"],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=torch.Tensor([0.4101762, 0.4342503, 0.3484594, 0.5473533]),
                std=torch.Tensor([0.17361328, 0.14048962, 0.12148701, 0.16887303]),
            ),
            K.Resize(224),
            data_keys=["image"],
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug or self.aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug or self.aug
            elif self.trainer.testing:
                aug = self.test_aug or self.aug
            elif self.trainer.predicting:
                aug = self.predict_aug or self.aug

            aug_batch = aug({"image": batch["image"].float()})

        return {
            "inputs": aug_batch["image"].float(),
            "targets": (batch["labels"].float() - 23.881872) / 31.52334,
        }


class USAVarsDataModuleOOD(NonGeoDataModule):
    """Adaptation for Data Module for OOD Experiments.

    Wrapper around TorchGeo Datamodule.

    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new instance of Data Module."""
        super().__init__(USAVarsOOD, batch_size, num_workers, **kwargs)

        # self.collate_fn = collate_fn_torchgeo

        Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
        self.aug: Transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )

    def ood_dataloader(
        self, ood_range: tuple[float, float]
    ) -> DataLoader[dict[str, Tensor]]:
        """Implement OOD Dataloader gicen the ood_range."""
        return DataLoader(
            dataset=self.dataset_class(split="ood", ood_range=ood_range, **self.kwargs),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug or self.aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug or self.aug
            elif self.trainer.testing:
                aug = self.test_aug or self.aug
            elif self.trainer.predicting:
                aug = self.predict_aug or self.aug

            aug_batch = aug({"image": batch["inputs"]})

        return {"inputs": aug_batch["image"], "targets": batch["targets"]}
