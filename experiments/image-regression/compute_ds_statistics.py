"""Compute dataset statistics from dataloader."""

from collections import defaultdict
from torch import Tensor
import torch
import numpy as np
from torchgeo.datasets import USAVars
from uq_method_box.datasets import USAVarsFeaturesOOD
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_statistics_on_batch(X: Tensor) -> Tensor:
    """Compute statistics on a batch of images.
    
    Args:
        X: input tensor of shape [batch_size, channels, height, width]
    
    Returns:
        channelwise statistics, keeping the batch_size
    """
    X = X.float() / 255
    batch_size = X.shape[0]
    flat_X = torch.flatten(X, start_dim=-2)
    min = flat_X.min(dim=-1).values
    max = flat_X.max(dim=-1).values
    mean = flat_X.mean(dim=-1)
    std = flat_X.std(dim=-1)
    height = X.shape[-2]
    width = X.shape[-1]
    return min.numpy(), max.numpy(), mean.numpy(), std.numpy(), np.ones(batch_size)*height*width
    

def compute_statistics_on_dl(dl: DataLoader) -> None:
    """Compute statistics on dataloader.
    
    Args:
        dl: dataloader 
    """
    out_dict = defaultdict(list)
    for batch in tqdm(dl):
        min, max, mean, std, px_count = compute_statistics_on_batch(batch["image"])
        out_dict["min"].append(min)
        out_dict["max"].append(max)
        out_dict["mean"].append(mean)
        out_dict["std"].append(std)
        out_dict["px_count"].append(px_count)
        out_dict["target"].append(batch["labels"].numpy() / 100)


    minimum = np.concatenate(out_dict["min"], axis=0).min(axis=0)
    maximum = np.concatenate(out_dict["max"], axis=0).max(axis=0)
    mean_vals = np.concatenate(out_dict["mean"], axis=0)
    std_vals = np.concatenate(out_dict["std"], axis=0)
    px_counts = np.concatenate(out_dict["px_count"], axis=0)

    targets = np.concatenate(out_dict["target"], axis=0)
    target_mean = targets.mean(0)
    target_std = targets.std(0)


    # minimum = np.amin(out[:, :, 0], axis=0)
    # maximum = np.amax(out[:, :, 1], axis=0)

    # mu_d = out[:, :, 2]
    mu = np.mean(mean_vals, axis=0)
    # sigma_d = out[:, :, 3]
    N_d = px_count[0]
    N = len(mean_vals) * N_d

    # https://stats.stackexchange.com/a/442050/188076
    sigma = np.sqrt(
        np.sum(std_vals**2 * (N_d - 1) + N_d * (mu - mean_vals) ** 2, axis=0) / (N - 1),
        dtype=np.float32,
    )

    np.set_printoptions(linewidth=2**8)
    print("min:", repr(minimum))
    print("max:", repr(maximum))
    print("mean:", repr(mu))
    print("std:", repr(sigma))
    print("target_mean:", repr(target_mean))
    print("target_std:", repr(target_std))


if __name__ == "__main__":
    ds = USAVarsFeaturesOOD(root="/mnt/SSD2/nils/uq-method-box/experiments/data/usa_vars", split="train", labels=["treecover"])
    dl = DataLoader(ds, batch_size=32)

    compute_statistics_on_dl(dl)