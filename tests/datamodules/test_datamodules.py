# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch.nn as nn
from lightning import Trainer

from lightning_uq_box.datamodules import (
    Toy8GaussiansDataModule,
    ToyDonutDataModule,
    ToyDUE,
    ToyGaussianSideWaysDataModule,
    ToySineDatamodule,
)
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeterministicRegression

# List of toy regression datasets
toy_datamodules = [
    ToySineDatamodule,
    ToyDonutDataModule,
    Toy8GaussiansDataModule,
    ToyGaussianSideWaysDataModule,
    ToyDUE,
]  # replace with actual dataset names


@pytest.mark.parametrize("datamodule", toy_datamodules)
def test_deterministic_regression_on_toy_datasets(datamodule, tmp_path: Path):
    # Initialize the data module
    data_module = datamodule()

    # Initialize the model
    model = DeterministicRegression(
        model=MLP(n_inputs=1, n_outputs=1), loss_fn=nn.MSELoss()
    )

    # Initialize the trainer
    trainer = Trainer(max_epochs=1, fast_dev_run=True, default_root_dir=tmp_path)

    # Fit and test
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
