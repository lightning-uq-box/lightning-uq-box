"""Tropical Cyclone OOD Dataset."""

import json
import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchgeo.datasets import TropicalCyclone
from tqdm import tqdm

# ds = TropicalCyclone(
#     "/home/nils/projects/uq-regression-box/experiments/data/tropicalCyclone"
# )

# fig, axs = plt.subplots(ncols=3)
# for idx, i in enumerate([57, 7862, 15687]):
#     sample = ds[i]
#     plt.sca(axs[idx])
#     ds.plot(sample, ax=axs[idx])

# import pdb

# pdb.set_trace()


# TODO
# 1. Come up with a useful dataset split
#    - across oceans ?
#    - across intensity? (maybe most suitable)
#    - across time (currently done, as task is time-series prediction task)
# 2. Precompute a "regression version" of our dataset with consecutively
#   sampled triplets of time steps to form RGB image
# 3. Should come up with a fixed dataset version for reproducibility
# (check the winning Data Driven solution)


class TropicalCycloneOOD(TropicalCyclone):
    """Tropical Cyclone Dataset adopted for OOD experiments."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new instance of the Dataset version."""
        super().__init__(root, split, None, download, api_key, checksum)

        # maybe get a collection of both train and test to do a reshuffle of train and val
        # could reshuffle the actual data on the first call and then just proceed as usual
        self.collection = self.retrieve_collection()

        # change self.collection to only include the images that we want
        # build a df for easier sampling and categorizing in groups
        collection_path = os.path.join(self.root, "collection_df.csv")
        if os.path.exists(collection_path):
            df = pd.read_csv(collection_path)
        else:
            items = []
            for item in tqdm(self.collection):
                source_id = item["href"].split("/")[0]
                if "train_source" in source_id:
                    split = "train"
                else:
                    split = "test"

                directory = os.path.join(
                    self.root,
                    "_".join([self.collection_id, split, "{0}"]),
                    source_id.replace("source", "{0}"),
                )
                features = self._load_features(directory)
                data = {"path": source_id, **features}
                # don't nee the tensor label, already includes 'wind_speed'
                data.pop("label")
                # but add original split where it is coming from
                data["split"] = split
                items.append(data)

            df = pd.DataFrame.from_dict(items)
            df.to_csv(collection_path)

        # ways to split
        # 1. across oceans
        # 2. across intensity
        # 3. across time (currently done, I think)

        # create some visualization statistics here
        # fig = self.create_summary_statistics(df)
        # fig = self.another_visualization(df)
        df.rename(columns={"Unnamed: 0": "full_df_idx"}, inplace=True)
        triplet_df = self.find_triplets(df)
        import pdb

        pdb.set_trace()

        # maybe find all possible triplet sequences for your dataset
        # and then randomly select some, could save this as a .json for reproducibility

        self.sample_collection = {0: ["paths"]}

        # adapt that you sample random sequences for three channel image like done in the challenge
        # since that really boosted results

    def create_summary_statistics(self, df: pd.DataFrame) -> plt.Figure:
        """Create summary statistics from dataset.

        Args:
            df: holds information about the collection for the desired split

        Returns:
            a matplotlib Figure showing summary statistics about the collection
        """
        fig, axs = plt.subplots(1, 4)

        df["storm_id"].value_counts().sort_values().plot(kind="bar", ax=axs[0])
        axs[0].set_title("Different storms and sequence length.")
        axs[0].set_xlabel("Storm ID")
        axs[0].set_ylabel("Number of images for this storm")

        df["ocean"].value_counts().sort_values().plot(kind="bar", ax=axs[1])
        axs[1].set_title("Ocean Distribution ")
        axs[1].set_xlabel("Ocean ID")
        axs[1].set_ylabel("Number of images for this ocean")

        storm_counts = df.groupby("storm_id").size()

        axs[2].hist(storm_counts, bins=50, color="lightgray")
        axs[2].set_title("Number of Images per Storm")
        axs[2].set_xlabel("Number of Images.")
        axs[2].set_ylabel("Number of Storms.")

        axs[3].violinplot(df["wind_speed"].values)
        axs[3].set_title("Distribution of Wind Speeds for entire dataset.")
        axs[3].set_xlabel("Wind Speed")

        return fig

    def another_visualization(self, df: pd.DataFrame):
        """"""
        fig, axs = plt.subplots(1, 3)
        max_wind_speeds = df.groupby("storm_id")["wind_speed"].max()

        axs[0].hist(max_wind_speeds, bins=50)
        axs[0].set_title("Max wind speed against number of storms.")
        axs[0].set_xlabel("Max wind speed.")
        axs[0].set_ylabel("Number of storms")

        # import pdb
        # pdb.set_trace()
        unique_ids = np.random.choice(df["storm_id"].unique(), 50)
        sampled_storms = df[df["storm_id"].isin(unique_ids)]

        for id in unique_ids:
            unique_df = sampled_storms[sampled_storms["storm_id"] == id]
            axs[1].plot(range(len(unique_df)), unique_df["wind_speed"], linewidth=0.5)

        axs[1].set_title(
            f"Wind speeds over time for {len(unique_ids)} randomly sampled storms."
        )
        axs[1].set_xlabel("Time.")
        axs[1].set_ylabel("Wind Speed.")

        axs[2] = sns.violinplot(data=df, x="ocean", y="wind_speed")
        axs[2].set_title("Distribution of wind speed across ocean.")
        axs[2].set_xlabel("Ocean Id")
        axs[2].set_ylabel("Wind speed distribution")
        return fig

    def find_triplets(self, df: pd.DataFrame):
        """Find triplet sequences of images per storm used as samples in deterministic way.

        Args:
            df: dataframe with complete annotations

        """
        unique_ids = df["storm_id"].unique().tolist()
        samples = []
        for id in unique_ids:
            unique_df = df[df["storm_id"] == id]

            candidates = unique_df[["wind_speed"]].rolling(window=3, step=3).max()

            # maybe criteria to drop some here

            # criteria to divide into train and test based on max_wind speed

            # collect triplet rgb image paths that belong to a single sample
            end_indices = candidates.index[1:]

            # define individual images that make up an RGB image sample
            sample_indices = [[idx - 2, idx - 1, idx] for idx in end_indices]

            # collect samples
            samples.extend(sample_indices)

        import pdb

        pdb.set_trace()
        print(0)

    def retrieve_collection(self) -> dict[str, Any]:
        """Retrieve collection from both train and val split."""
        output_dir = "_".join([self.collection_id, "train", "source"])
        filename = os.path.join(self.root, output_dir, "collection.json")
        with open(filename) as f:
            self.train_collection = json.load(f)["links"]

        output_dir = "_".join([self.collection_id, "test", "source"])
        filename = os.path.join(self.root, output_dir, "collection.json")
        with open(filename) as f:
            self.test_collection = json.load(f)["links"]

        return self.train_collection + self.test_collection

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        paths = self.sample_collection[index]

        source_id = self.collection[index]["href"][0]
        directory = os.path.join(
            self.root,
            "_".join([self.collection_id, self.split, "{0}"]),
            source_id.replace("source", "{0}"),
        )

        sample: dict[str, Any] = {
            "image": torch.stack([self._load_image(paths)], dim=-1)
        }

        features = [self._load_features(paths)]

        # get correct feature

        sample.update(**features)

        return sample


ds = TropicalCycloneOOD(
    "/home/nils/projects/uq-method-box/experiments/data/tropicalCyclone"
)


unique_ids = df["storm_id"].unique().tolist()
samples = []
for id in unique_ids:
    unique_df = df[df["storm_id"] == id]

    candidates = unique_df[["wind_speed"]].rolling(window=3, step=3).max()

    # maybe criteria to drop some here

    # criteria to divide into train and test based on max_wind speed

    # collect triplet rgb image paths that belong to a single sample
    end_indices = candidates.index[1:]

    # define individual images that make up an RGB image sample
    sample_indices = [[idx - 2, idx - 1, idx] for idx in end_indices]

    # collect samples
    samples.extend(sample_indices)
