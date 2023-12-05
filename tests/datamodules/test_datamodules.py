# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

import pytest
from lightning import Trainer
import torch.nn as nn
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeterministicRegression
from lightning_uq_box.datamodules import ToyGaussianSideWaysDataModule, ToyDonutDataModule, Toy8GaussiansDataModule, ToySineDatamodule

# List of toy regression datasets
toy_datamodules = [ToySineDatamodule, ToyDonutDataModule, Toy8GaussiansDataModule, ToyGaussianSideWaysDataModule]  # replace with actual dataset names

@pytest.mark.parametrize("datamodule", toy_datamodules)
def test_deterministic_regression_on_toy_datasets(datamodule):
    # Initialize the data module
    data_module = datamodule()

    # Initialize the model
    model = DeterministicRegression(
        model=MLP(n_inputs=1, n_outputs=1),
        loss_fn=nn.MSELoss(),
    )

    # Initialize the trainer
    trainer = Trainer(max_epochs=1, fast_dev_run=True)

    # Fit and test
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)