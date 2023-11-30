# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: uqboxEnv
#     language: python
#     name: python3
# ---

# # Deep Kernel Learning Classification

# ## Imports

# +
import os
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import VariationalELBO
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (  # BaseModel,; DeterministicGaussianModel,
    DKLClassification,
    DKLGPLayer,
)
from lightning_uq_box.viz_utils import (
    plot_predictions_classification,
    plot_training_metrics,
    plot_two_moons_data,
)

# -

seed_everything(2)

# temporary directory for saving
my_temp_dir = tempfile.mkdtemp()

dm = TwoMoonsDataModule(batch_size=100)

# define data
X_train, y_train, X_test, y_test, test_grid_points = (
    dm.X_train,
    dm.y_train,
    dm.X_test,
    dm.y_test,
    dm.test_grid_points,
)

plot_two_moons_data(X_train, y_train, X_test, y_test)

# ## Feature Extractor

feature_extractor = MLP(
    n_inputs=2, n_outputs=13, n_hidden=[50], activation_fn=torch.nn.ELU()
)

# ## Deep Kernel Learning Model

dkl_model = DKLClassification(
    feature_extractor,
    gp_kernel="RBF",
    num_classes=2,
    optimizer=partial(torch.optim.Adam, lr=1e-2),
    n_inducing_points=100,
)

# ## Trainer

logger = CSVLogger(my_temp_dir)
trainer = Trainer(
    max_epochs=100,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=1,
    enable_checkpointing=False,
    enable_progress_bar=False,
)

trainer.fit(dkl_model, dm)

# ## Training Metrics

fig = plot_training_metrics(os.path.join(my_temp_dir, "lightning_logs"), "Acc")

# ## Prediction

# save predictions
trainer.test(dkl_model, dm.test_dataloader())

# ## Evaluate Predictions

#
