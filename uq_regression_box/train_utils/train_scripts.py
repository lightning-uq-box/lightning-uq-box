"""Simple training scripts for UQ-methods."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange


def basic_train_loop(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    lr: float,
    n_epochs: int = 100,
):
    """Train model for Map estimate.

    Args:
      model: model to train
      criterion: loss function
      train_loader: dataloder with training data
      lr: learning rate
      n_epochs: number of epochs to train for

    Returns:
      trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bar = trange(n_epochs)
    for i in bar:
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.detach().cpu().item()}")
    return model
