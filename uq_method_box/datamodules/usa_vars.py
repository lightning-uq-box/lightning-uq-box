"""USA Vars datamodule adaption for OOD experiments."""

from typing import Any, Callable, Dict

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule, USAVarsDataModule, USAVarsFeatureExtractedDataModule
from torchgeo.transforms import AugmentationSequential

from uq_method_box.datasets import USAVarsOOD, USAVarsFeaturesOOD, USAVarsFeaturesOur


class USAVarsFeatureExtractedDataModuleOur(NonGeoDataModule):
    """USAVarsFeatureExtracted Data Module."""

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Version we use for now."""
        super().__init__(USAVarsFeaturesOur, batch_size, num_workers, **kwargs)
        ds = self.dataset_class(**kwargs, split="train")
        if ds.feature_extractor == "rcf_8192":
            feature_cols = [str(i) for i in range(8192)]
        else:
            feature_cols = [str(i) for i in range(512)]
        feature_df = ds.feature_df
        self.input_mean = torch.from_numpy(feature_df[feature_cols].mean().values).to(torch.float)
        self.input_std = torch.from_numpy(feature_df[feature_cols].std().values).to(torch.float)
        self.target_mean = feature_df["treecover"].mean()
        self.target_std = feature_df["treecover"].std()

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
        if self.input_mean.device != batch["image"].device:
            if self.input_mean.device.type == "cpu":
                self.input_mean = self.input_mean.to(batch["image"].device)
                self.input_std = self.input_std.to(batch["image"].device)
            elif self.input_mean.device.type == "cuda":
                batch["image"] = batch["image"].to(self.input_mean.device)
                batch["labels"] = batch["labels"].to(self.input_mean.device)

        new_batch = {
            "inputs": (batch["image"].float() - self.input_mean) / self.input_std,
            "targets": (batch["labels"].float() - self.target_mean) / self.target_std
        }

        if "centroid_lat" in batch:
            new_batch["centroid_lat"] = batch["centroid_lat"]
            new_batch["centroid_lon"] = batch["centroid_lon"]
        
        return new_batch

class USAVarsFeatureExtractedDataModuleOOD(NonGeoDataModule):
    def  __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Version we use for now for OOD."""
        super().__init__(USAVarsFeaturesOOD, batch_size, num_workers, **kwargs)
        ds = self.dataset_class(**kwargs, split="train")
        if ds.feature_extractor == "rcf_8192":
            feature_cols = [str(i) for i in range(8192)]
        else:
            feature_cols = [str(i) for i in range(512)]
        feature_df = ds.feature_df
        self.input_mean = torch.from_numpy(feature_df[feature_cols].mean().values).to(torch.float)
        self.input_std = torch.from_numpy(feature_df[feature_cols].std().values).to(torch.float)
        self.target_mean: float = feature_df["treecover"].mean()
        self.target_std: float = feature_df["treecover"].std()

    def ood_dataloader(
        self, ood_range: tuple[float, float] = None
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
        if self.input_mean.device != batch["image"].device:
            if self.input_mean.device.type == "cpu":
                self.input_mean = self.input_mean.to(batch["image"].device)
                self.input_std = self.input_std.to(batch["image"].device)
            elif self.input_mean.device.type == "cuda":
                batch["image"] = batch["image"].to(self.input_mean.device)
                batch["labels"] = batch["labels"].to(self.input_mean.device)
        new_batch = {
            "inputs": (batch["image"].float() - self.input_mean) / self.input_std,
            "targets": (batch["labels"].float() - self.target_mean) / self.target_std
        }

        if "centroid_lat" in batch:
            new_batch["centroid_lat"] = batch["centroid_lat"]
            new_batch["centroid_lon"] = batch["centroid_lon"]
        
        return new_batch

class USAVarsDataModuleOur(USAVarsDataModule):
    # min: array([0., 0., 0., 0.], dtype=float32)
    # max: array([1., 1., 1., 1.], dtype=float32)
    # mean: array([0.4101762, 0.4342503, 0.3484594, 0.5473533], dtype=float32)
    # std: array([0.17361328, 0.14048962, 0.12148701, 0.16887303], dtype=float32)
    # target_mean: array([0.23881873], dtype=float32) target range 0-1 and then normalize
    # target_std: array([0.31523344], dtype=float32)
    """USA Vars Datamodule."""

    # target_mean: array([23.881872], dtype=float32)
    # target_std: array([31.52334], dtype=float32)

    input_mean = torch.Tensor([0.4101762, 0.4342503, 0.3484594, 0.5473533])
    input_std = torch.Tensor([0.17361328, 0.14048962, 0.12148701, 0.16887303])
    target_mean = 23.881872
    target_std = 31.52334

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Version we use for now."""
        super().__init__(batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=self.input_mean,
                std=self.input_std,
            ),
            K.Resize(224),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image"],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=self.input_mean,
                std=self.input_std,
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

        new_batch = {
            "inputs": (batch["image"].float() - self.input_mean) / self.input_std,
            "targets": (batch["labels"].float() - self.target_mean) / self.target_std
        }

        if "centroid_lat" in batch:
            new_batch["centroid_lat"] = batch["centroid_lat"]
            new_batch["centroid_lon"] = batch["centroid_lon"]
        
        return new_batch


class USAVarsDataModuleOOD(NonGeoDataModule):
    # min: array([0., 0., 0., 0.], dtype=float32)
    # max: array([1., 1., 1., 1.], dtype=float32)
    # mean: array([0.45211497, 0.45899174, 0.3701368 , 0.5534093], dtype=float32)
    # std: array([ ], dtype=float32)
    # target_mean: array([6.022064], dtype=float32)
    # target_std: array([10.314759], dtype=float32)

    """Adaptation for Data Module for OOD Experiments.

    Wrapper around TorchGeo Datamodule.

    """
    input_mean = torch.Tensor([0.45211497, 0.45899174, 0.3701368 , 0.5534093])
    input_std = torch.Tensor([0.16486272, 0.13277882, 0.11872848, 0.1632025])
    target_mean = 6.022064
    target_std = 10.314759

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new instance of Data Module."""
        super().__init__(USAVarsOOD, batch_size, num_workers, **kwargs)

        # self.collate_fn = collate_fn_torchgeo

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=self.input_mean,
                std=self.input_std,
            ),
            K.Resize(224),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image"],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(
                mean=self.input_mean,
                std=self.input_std,
            ),
            K.Resize(224),
            data_keys=["image"],
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

        new_batch = {
            "inputs": (batch["image"].float() - self.input_mean) / self.input_std,
            "targets": (batch["labels"].float() - self.target_mean) / self.target_std
        }

        if "centroid_lat" in batch:
            new_batch["centroid_lat"] = batch["centroid_lat"]
            new_batch["centroid_lon"] = batch["centroid_lon"]
        
        return new_batch
