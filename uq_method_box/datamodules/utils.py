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

    # Create the new batch dictionary with keys "inputs" and "targets"
    new_batch = {"inputs": inputs, "targets": targets}

    return new_batch
