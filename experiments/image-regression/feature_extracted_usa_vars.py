"""Script to create feature extracted version of USA Vars."""

import argparse

import kornia.augmentation as K
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import USAVars
from torchgeo.models import RCF
from torchgeo.transforms import AugmentationSequential
from tqdm import tqdm


def get_feature_extractor(feature_extractor: str) -> torch.nn.Module:
    """Configure the feature extractor."""
    if feature_extractor == "rcf":
        model = RCF(in_channels=4, features=512, kernel_size=3, seed=0)
    elif feature_extractor == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    else:
        raise ValueError
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="root directory for USA Vars dataset")
    parser.add_argument("--save_dir", help="where to save the created dataset")
    parser.add_argument(
        "--feature_extractor",
        default="rcf",
        help="which feature extractor to use",
        choices=["rcf", "resnet18"],
    )
    args = parser.parse_args()

    aug = AugmentationSequential(
        K.Normalize(mean=torch.zeros(4), std=torch.ones(4) * 255),
        K.Normalize(
            mean=torch.Tensor([0.4101762, 0.4342503, 0.3484594, 0.5473533]),
            std=torch.Tensor([0.17361328, 0.14048962, 0.12148701, 0.16887303]),
        ),
        K.Resize(224),
        data_keys=["image"],
    )

    feature_extractor = get_feature_extractor(args.feature_extractor)

    df = pd.DataFrame()

    for split in ["train", "val", "test"]:
        ds = USAVars(args.root, split=split)

        dl = DataLoader(ds, batch_size=16, num_workers=4)

        split_features = []
        split_targets = []
        filenames = []
        centroid_lat = []
        centroid_lon = []
        for batch in tqdm(dl):
            with torch.no_grad():
                aug_batch = aug(
                    {"image": batch["image"].float(), "labels": batch["labels"]}
                )
                features = feature_extractor(aug_batch["image"]).numpy()
            targets = aug_batch["labels"].numpy()

            split_features.append(features)
            split_targets.append(targets)
            filenames.extend(batch["filename"])
            centroid_lat.append(batch["centroid_lat"])
            centroid_lon.append(batch["centroid_lon"])

        split_features = np.concatenate(split_features)
        split_targets = np.concatenate(split_targets)
        centroid_lon = np.concatenate(centroid_lon)
        centroid_lat = np.concatenate(centroid_lat)

        split_df = pd.DataFrame(split_features)
        split_df["centroid_lat"] = centroid_lat
        split_df["centroid_lon"] = centroid_lon
        split_df["filename"] = filenames
        split_df["split"] = split
        for idx, label in enumerate(ds.labels):
            split_df[label] = targets[:, idx]

        df = pd.concat([df, split_df])

    # save as csv
    df.to_csv(args.save_dir, index=False)
