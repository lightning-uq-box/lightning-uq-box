"""Utility functions for datamodules."""

import torch


def collate_fn_torchgeo(batch):
    """Collate function to change torchgeo naming conventions to ours.

    Args:
        batch: input batch

    Returns:
        renamed batch
    """
    # Extract images and labels from the batch dictionary
    images = [item["image"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Stack images and labels into tensors
    inputs = torch.stack(images)
    targets = torch.stack(labels)

    return {"inputs": inputs, "targets": targets}


def collate_fn_tensordataset(batch):
    """Collate function for tensor dataset to our framework."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return {"inputs": inputs, "targets": targets}


def collate_fn_laplace_torch(batch):
    """Collate function to for laplace torch tuple convention.

    Args:
        batch: input batch

    Returns:
        renamed batch
    """
    # Extract images and labels from the batch dictionary
    try:
        images = [item["image"] for item in batch]
        labels = [item["labels"] for item in batch]
    except KeyError:
        images = [item["inputs"] for item in batch]
        labels = [item["targets"] for item in batch]

    # Stack images and labels into tensors
    inputs = torch.stack(images)
    targets = torch.stack(labels)

    return (inputs, targets)
