# ---
# jupyter:
#   jupytext:
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

# # MC-Dropout Classification

# ## Imports

# +
import os
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch.optim import Adam

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import MCDropoutClassification
from lightning_uq_box.viz_utils import (
    plot_predictions_classification,
    plot_training_metrics,
    plot_two_moons_data,
)

plt.rcParams["figure.figsize"] = [14, 5]

# %load_ext autoreload
# %autoreload 2
# -

seed_everything(0)

my_temp_dir = tempfile.mkdtemp()

# ## Datamodule

# +
dm = TwoMoonsDataModule(batch_size=128)

X_train, y_train, X_test, y_test, test_grid_points = (
    dm.X_train,
    dm.y_train,
    dm.X_test,
    dm.y_test,
    dm.test_grid_points,
)
# -

plot_two_moons_data(X_train, y_train, X_test, y_test)

# ## Model

network = MLP(
    n_inputs=2,
    n_hidden=[50, 50, 50],
    n_outputs=2,
    dropout_p=0.2,
    activation_fn=nn.ReLU(),
)
network

mc_dropout_module = MCDropoutClassification(
    model=network,
    optimizer=partial(Adam, lr=1e-2),
    loss_fn=nn.CrossEntropyLoss(),
    num_mc_samples=25,
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

trainer.fit(mc_dropout_module, dm)

# ## Training Metrics

fig = plot_training_metrics(os.path.join(my_temp_dir, "lightning_logs"), "Acc")

# ## Prediction

# save predictions
trainer.test(mc_dropout_module, dm.test_dataloader())

preds = mc_dropout_module.predict_step(test_grid_points)

# ## Evaluate Predictions

fig = plot_predictions_classification(
    X_test, y_test, preds["pred"].argmax(-1), test_grid_points, preds["pred_uct"]
)

#
