import os
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from uq_method_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToySineDatamodule,
    ToyUncertaintyGaps,
)
from uq_method_box.models import MLP
from uq_method_box.uq_methods import (
    NLL,
    DeterministicGaussianModel,
    MCDropoutModel,
    SGLDModel,
)
from uq_method_box.viz_utils import plot_predictions

seed_everything(4)

dm = ToyUncertaintyGaps()

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

max_epochs = 500
my_config = {
    "model_args": {"n_inputs": 2, "n_outputs": 2, "n_hidden": [50]},
    "loss_fn": "nll",
    "n_sgld_samples": int(0.75 * max_epochs),
    "burnin_epochs": max_epochs - int(0.75 * max_epochs),
}

my_dir = tempfile.mkdtemp()

base_model = DeterministicGaussianModel(
    MLP(**my_config["model_args"]),
    optimizer=partial(torch.optim.Adam, lr=3e-3),
    max_epochs=max_epochs,
    burnin_epochs=10,
    save_dir=my_dir,
)

sgld_model = SGLDModel(
    base_model.model,
    NLL(),
    train_loader=train_loader,
    save_dir=my_dir,
    lr=2e-4,
    n_sgld_samples=my_config["n_sgld_samples"],
    burnin_epochs=my_config["burnin_epochs"],
    part_stoch_module_names=[-1],
    max_epochs=max_epochs,
    weight_decay=0.0,
    noise_factor=0.01,
)

logger = CSVLogger(my_dir)


# fit model
# train base model
trainer = Trainer(max_epochs=max_epochs, logger=logger, log_every_n_steps=10)
trainer.fit(base_model, dm)

# post-process sgld model
trainer = Trainer(max_epochs=max_epochs, logger=logger, log_every_n_steps=1)
# using the trainer does save predictions
trainer.test(sgld_model, dm.test_dataloader())
csv_path = os.path.join(my_dir, "predictions.csv")

# this does not save any predictions because
# lightning hooks are not called, only invoking single method
# pred = model.test_step((X_test, y_test))
# pred = model.predict_step(X_test)

with torch.no_grad():
    out = sgld_model.predict_step(X_test)

sgld_fig = plot_predictions(
    dm.X_train_plot,
    y_train,
    dm.X_test_plot,
    y_test,
    out["mean"],
    out["pred_uct"],
    epistemic=out.get("epistemic_uct", None),
    aleatoric=out.get("aleatoric_uct", None),
    title="SGLD",
)

# metrics_path = os.path.join(my_dir, "lightning_logs", "version_0", "metrics.csv")
# df = pd.read_csv(metrics_path)

# train_loss = df[df["train_loss"].notna()]["train_loss"]
# train_rmse = df[df["train_RMSE"].notna()]["train_RMSE"]

# fig, ax = plt.subplots(2)
# ax[0].plot(np.arange(len(train_loss)), train_loss)
# ax[0].set_title("Train Loss")

# ax[1].plot(np.arange(len(train_rmse)), train_rmse)
# ax[1].set_title("Train RMSE")

import pdb

pdb.set_trace()
print(0)
