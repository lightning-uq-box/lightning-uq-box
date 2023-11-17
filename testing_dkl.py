"""Test DKL implementation."""

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

from lightning_uq_box.datamodules import (
    ToyDUE,
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (  # BaseModel,; DeterministicGaussianModel,
    DKLClassification,
    DKLGPLayer,
    DKLRegression,
    DUEClassification,
)
from lightning_uq_box.viz_utils import (
    plot_predictions_classification,
    plot_predictions_regression,
)

seed_everything(2)

# define datamodule

dm = TwoMoonsDataModule(batch_size=100)

# define data
X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

# temporary directory for saving
my_dir = tempfile.mkdtemp()

# define untrained feature extractor
feature_extractor = MLP(
    n_inputs=2, n_outputs=13, n_hidden=[50], activation_fn=torch.nn.ELU()
)


dkl_model = DUEClassification(
    feature_extractor,
    gp_layer=partial(DKLGPLayer, n_outputs=2, kernel="Matern52"),
    elbo_fn=partial(VariationalELBO),
    optimizer=partial(torch.optim.Adam, lr=1e-2),
    n_inducing_points=85,
    input_size=2,
)

max_epochs = 100

logger = CSVLogger(my_dir)

pl_args = {
    "max_epochs": max_epochs,
    "logger": logger,
    "log_every_n_steps": 1,
    "accelerator": "cpu",
    "limit_val_batches": 0.0,
    # "devices": [0],
}
trainer = Trainer(**pl_args)

# fit model
trainer.fit(dkl_model, dm)

# using the trainer does save predictions
trainer.test(dkl_model, dm.test_dataloader())
csv_path = os.path.join(my_dir, "predictions.csv")


if isinstance(dkl_model, DKLRegression):
    with torch.no_grad():
        pred = dkl_model.predict_step(X_test)
    dkl_fig = plot_predictions_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        pred["pred"],
        pred["pred_uct"],
        epistemic=pred["epistemic_uct"],
        aleatoric=pred.get("aleatoric_uct", None),
        title="DUE",
    )
else:
    with torch.no_grad():
        pred = dkl_model.predict_step(dm.test_grid_points)
    dkl_fig = plot_predictions_classification(
        X_test=X_test,
        y_test=y_test,
        y_pred=pred["pred"].argmax(-1),
        pred_uct=pred["pred_uct"],
        test_grid_points=dm.test_grid_points,
    )

plt.savefig("dkl.png")

import pdb

pdb.set_trace()

metrics_path = os.path.join(my_dir, "lightning_logs", "version_0", "metrics.csv")
df = pd.read_csv(metrics_path)

train_loss = df[df["train_loss"].notna()]["train_loss"]
train_rmse = df[df["train_RMSE"].notna()]["train_RMSE"]

fig, ax = plt.subplots(2)
ax[0].plot(np.arange(len(train_loss)), train_loss)
ax[0].set_title("Train Loss")

ax[1].plot(np.arange(len(train_rmse)), train_rmse)
ax[1].set_title("Train RMSE")

plt.savefig("dkl_loss_curve.png")


plt.show()

# print(0)
