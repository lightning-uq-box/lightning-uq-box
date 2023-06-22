"""USA Vars Dataset for OOD Tasks."""

import copy
import os
from typing import Dict, Optional, Sequence

import numpy as np
from torch import Tensor
from torchgeo.datasets import USAVars

# class USAVarsOur(USAVars):
#     def __init__(
#         self,
#         root: str = "data",
#         split: str = "train",
#         labels: Sequence[str] = ["treecover"],
#         download: bool = False,
#         checksum: bool = False,
#     ) -> None:
#         """Initialize a new USAVars dataset instance.

#         Args:
#             root: root directory where dataset can be found
#             split: train/val/test split to load in distribution sets, `ood`
#                 for out-of distribution set
#             labels: list of labels to include
#             transforms: a function/transform that takes input sample and its target as
#                 entry and returns a transformed version
#             download: if True, download dataset and store it in the root directory
#             checksum: if True, check the MD5 of the downloaded files (may be slow)

#         Raises:
#             AssertionError: if invalid labels are provided
#             ImportError: if pandas is not installed
#             RuntimeError: if ``download=False`` and data is not found, or checksums
#                 don't match
#         """
#         super().__init__(root, split, labels, None, download, checksum)


class USAVarsOOD(USAVars):
    """USA Vars Dataset adapted for OOD."""

    valid_splits = ["train", "val", "test", "ood"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        # labels: Sequence[str] = ["treecover"],
        ood_range: Optional[tuple[float, float]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load in distribution sets, `ood`
                for out-of distribution set
            labels: list of labels to include
            ood_range: range of target values which to consider for ood split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if invalid labels are provided
            ImportError: if pandas is not installed
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        # root = '/home/user/uq-method-box/experiments/data/usa_vars'
        super().__init__(root, "train", ["treecover"], None, download, checksum)

        self.availabel_files = copy.deepcopy(self.files)

        assert (
            split in self.valid_splits
        ), f"Valid splits are {self.valid_splits}, but found {split}."
        if split == "ood" and not ood_range:
            raise ValueError("Need to specify `ood_range`.")

        self.label_df = self.label_dfs["treecover"]

        # there are more ids in label df present than files included
        self.file_ids = [file.split("_")[1].split(".")[0] for file in self.files]
        self.label_df = self.label_df.loc[self.file_ids]

        self.in_dist_set = self.label_df[self.label_df["treecover"] <= 10]

        if split == "ood":
            assert (
                len(ood_range) == 2
            ), "Please only specify a min and max range value in that order."
            assert (
                ood_range[0] < ood_range[1]
            ), "Please first specify the min and then the max range value."
            self.ood_range = ood_range
            self.ood_set = self.label_df[
                (self.label_df["treecover"] > ood_range[0])
                & (self.label_df["treecover"] <= ood_range[1])
            ]
            self.ood_ids = self.ood_set.index.values

        np.random.seed(0)
        in_dist_file_ids = self.in_dist_set.index.values
        split1 = int(0.7 * len(in_dist_file_ids))
        split2 = int(0.85 * len(in_dist_file_ids))

        np.random.shuffle(in_dist_file_ids)
        self.train_ids = in_dist_file_ids[:split1]
        self.val_ids = in_dist_file_ids[split1:split2]
        self.test_ids = in_dist_file_ids[split2:]

        if split == "train":
            self.files = [f"tile_{id}.tif" for id in self.train_ids]
        elif split == "val":
            self.files = [f"tile_{id}.tif" for id in self.val_ids]
        elif split == "test":
            self.files = [f"tile_{id}.tif" for id in self.test_ids]
        else:  # ood
            self.files = [f"tile_{id}.tif" for id in self.ood_ids]

    def _load_files(self) -> list[str]:
        """Load all files."""
        all_files = []
        for split in self.split_metadata.keys():
            with open(os.path.join(self.root, f"{split}_split.txt")) as f:
                files = f.read().splitlines()
                all_files.extend(files)
        return all_files

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample = super().__getitem__(index)
        return {"inputs": sample["image"].float(), "targets": sample["labels"].float()}

    def plot_geo_distribution(self):
        """Plot geo distribution."""
        import geopandas
        import geoplot as gplt

        treecover_gdf = geopandas.GeoDataFrame(
            self.label_df,
            geometry=geopandas.points_from_xy(self.label_df.lon, self.label_df.lat),
        )

        n = 20000
        ax = gplt.pointplot(
            treecover_gdf.sample(n), hue="treecover", legend=True, figsize=(16, 12)
        )

        ax.set_title(
            f"Spatial Distribution of Treecover for {n} randomly sampled points."
        )


ds = USAVarsOOD(
    root="/mnt/SSD2/nils/uq-method-box/experiments/data/usa_vars",
    split="ood",
    ood_range=[20, 30]
    # download=True
)
