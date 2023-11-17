import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeterministicClassification, SWAGClassification
from lightning_uq_box.viz_utils import (
    plot_predictions_classification,
    plot_training_metrics,
    plot_two_moons_data,
)

plt.rcParams["figure.figsize"] = [14, 5]

max_epochs = 100

dm = TwoMoonsDataModule()

X_train, y_train, X_test, y_test, test_grid_points = (
    dm.X_train,
    dm.y_train,
    dm.X_test,
    dm.y_test,
    dm.test_grid_points,
)

network = MLP(n_inputs=2, n_hidden=[50, 50, 50], n_outputs=2, activation_fn=nn.ReLU())

my_dir = tempfile.mkdtemp()


deterministic_model = DeterministicClassification(
    network, partial(torch.optim.Adam, lr=1e-3), loss_fn=nn.CrossEntropyLoss()
)

trainer = Trainer(max_epochs=max_epochs, logger=False)

# fit model
trainer.fit(deterministic_model, dm)


# fit laplace
swag_model = SWAGClassification(
    deterministic_model.model,
    num_swag_epochs=30,
    max_swag_snapshots=30,
    snapshot_freq=1,
    num_mc_samples=50,
    swag_lr=1e-4,
    loss_fn=nn.CrossEntropyLoss(),
)

trainer = Trainer(enable_progress_bar=False)
trainer.test(swag_model, datamodule=dm)

preds = swag_model.predict_step(test_grid_points)

fig = plot_predictions_classification(
    X_test, y_test, preds["pred"].argmax(-1), test_grid_points, preds["pred_uct"]
)

import pdb

pdb.set_trace()

print(0)
