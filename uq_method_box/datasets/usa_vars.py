"""USA Vars Dataset for OOD Tasks."""

import copy
import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from torch import Tensor
from torchgeo.datasets import USAVars, USAVarsFeatureExtracted

class USAVarsFeaturesOur(USAVarsFeatureExtracted):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels: Sequence[str] = ["treecover"],
        feature_extractor: str = "rcf",
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load in distribution sets, `ood`
                for out-of distribution set
            labels: list of labels to include
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
        super().__init__(root, split, labels, feature_extractor, download)

        # num_total = len(self.feature_df[self.feature_df["treecover"]<10])
        # discard = self.feature_df[self.feature_df["treecover"]<10].sample(n=num_total-5000)
        # self.feature_df = self.feature_df.drop(discard.index).reset_index(drop=True)
        # import pdb
        # pdb.set_trace()


    

class USAVarsFeaturesOOD(USAVarsFeatureExtracted):

    valid_splits = ["train", "val", "test", "ood"]
    in_target_max = 40
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels: Sequence[str] = ["treecover"],
        ood_type: str = "tail",
        ood_range: Optional[tuple[float, float]] = None,
        feature_extractor: str = "rcf",
        download: bool = False,
    ) -> None:
        print(ood_type)
        super().__init__(root, split, ["treecover"], feature_extractor, download)

        self.full_df = pd.read_csv(
            os.path.join(self.root, self.csv_file_name.format(self.feature_extractor))
        )

        self.ood_type = ood_type

        assert (
            split in self.valid_splits
        ), f"Valid splits are {self.valid_splits}, but found {split}."

        if split == "ood":
            if ood_type == "tail":
                assert (
                    ood_range[0] < ood_range[1]
                ), "Please first specify the min and then the max range value."
                assert (ood_range[0] >= self.in_target_max), "OOD min should be larger than in distribution max."
                self.ood_range = ood_range


        # rebalance
        if self.ood_type == "tail":
            self.in_dist_df = self.full_df[self.full_df["treecover"] < self.in_target_max].reset_index(drop=True)
        else:
            self.in_dist_df = self.full_df[(self.full_df["treecover"] < 33.333) | (self.full_df["treecover"] > 66.666)].reset_index(drop=True)
        num_total = len(self.in_dist_df[self.in_dist_df["treecover"]<10])
        discard = self.in_dist_df[self.in_dist_df["treecover"]<10].sample(n=num_total-5000)
        self.in_dist_df = self.in_dist_df.drop(discard.index).reset_index(drop=True)



        # ood splits
        if self.ood_type == "tail":
            self.ood_df = self.full_df[self.full_df["treecover"] >= self.in_target_max].reset_index(drop=True)
        else:
            self.ood_df = self.full_df[(self.full_df["treecover"] >= 33.333) & (self.full_df["treecover"] <= 66.666)].reset_index(drop=True)

        # from sklearn.metrics.pairwise import cosine_similarity
        # import matplotlib.pyplot as plt
        # feature_cols = [str(i) for i in range(512)]
        # np.random.seed(0)
        # for ood_range in [(40, 60), (60, 80), (80, 100)]:
        #     ood_set = self.ood_df[(self.ood_df["treecover"]>=ood_range[0]) & (self.ood_df["treecover"]<=ood_range[1])]
        #     in_samples = self.in_dist_df.sample(n=5000, weights=self.in_dist_df["treecover"]).sort_values(by="treecover", ascending=True)

        #     in_target = in_samples["treecover"].values
        #     in_features = in_samples[feature_cols].values

        #     ood_samples = ood_set.sample(n=5000).sort_values(by="treecover", ascending=True)
        #     out_target = ood_samples["treecover"].values
        #     out_features = ood_samples[feature_cols].values

        #     sim_matrix = cosine_similarity(in_features, out_features)
        #     fig, ax = plt.subplots(1)
        #     img = ax.imshow(sim_matrix, cmap="plasma")
        #     fig.colorbar(img, ax=ax)
        #     ax.set_xlabel("OOD SAMPLES")
        #     ax.set_ylabel("IN SAMPLES")
        #     # pick x ticks
        #     x_tick_idx = np.linspace(0, len(out_target)-1, 5, dtype=int)
        #     x_ticks = [round(out_target[idx], 3) for idx in x_tick_idx]

        #     y_tick_idx = np.linspace(0, len(in_target)-1, 5, dtype=int)
        #     y_ticks = [round(in_target[idx], 3) for idx in y_tick_idx]
        #     plt.xticks(x_tick_idx, x_ticks)
        #     plt.yticks(y_tick_idx, y_ticks)
        #     ax.set_title(f"Pairwise feature Cosine similary of ordered samples {ood_range[0]}_{ood_range[1]}")
        #     plt.savefig(f"cosine_{ood_range[0]}_{ood_range[1]}")
        # import pdb
        # pdb.set_trace()
        # import matplotlib.pyplot as plt
        


        # self.active_learn_df = self.ood_df.sample(frac=0.4)
        # self.ood_df = self.ood_df.drop(self.active_learn_df.index)
        # self.in_dist_df = pd.concat([self.in_dist_df, self.active_learn_df], ignore_index=True).reset_index(drop=True)

        in_dist_ids = self.in_dist_df.index.values
        split1 = int(0.7 * len(in_dist_ids))
        split2 = int(0.85 * len(in_dist_ids))

        np.random.shuffle(in_dist_ids)
        self.train_ids = in_dist_ids[:split1]
        self.val_ids = in_dist_ids[split1:split2]
        self.test_ids = in_dist_ids[split2:]

        if split == "train":
            self.feature_df = self.in_dist_df[self.in_dist_df.index.isin(self.train_ids)]
        elif split == "val":
            self.feature_df = self.in_dist_df[self.in_dist_df.index.isin(self.val_ids)]
        elif split == "test":
            self.feature_df = self.in_dist_df[self.in_dist_df.index.isin(self.test_ids)]
        else:  # ood
            if self.ood_type == "tail":
                self.feature_df = self.ood_df[(self.ood_df["treecover"]>=ood_range[0]) & (self.ood_df["treecover"]<=ood_range[1])]
            else:
                self.feature_df = self.ood_df.copy()

        # reset index for length of dataset and proper indexing
        self.feature_df.reset_index(inplace=True)

class USAVarsOOD(USAVars):
    """USA Vars Dataset adapted for OOD."""

    valid_splits = ["train", "val", "test", "ood"]
    in_target_max = 40
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels: Sequence[str] = ["treecover"],
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

        self.in_dist_set = self.label_df[self.label_df["treecover"] < self.in_target_max]

        if split == "ood":
            assert (
                len(ood_range) == 2
            ), "Please only specify a min and max range value in that order."
            assert (
                ood_range[0] < ood_range[1]
            ), "Please first specify the min and then the max range value."
            assert (ood_range[0] >= in_target_max), "OOD min should be larger than in distribution max."
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