"""BNN+VI Testing Script Toy Data."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BNN_VI
from uq_method_box.viz_utils import plot_predictions

# seed_everything(4)
torch.set_float32_matmul_precision("medium")


dm = ToyHeteroscedasticDatamodule(batch_size=100)

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

my_config = {
    "model_args": {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_hidden": [100, 100],
        "activation_fn": torch.nn.Tanh(),
    },
    "loss_fn": "nll",
}

my_dir = tempfile.mkdtemp()

max_epochs = 1000

base_model = BNN_VI(
    MLP,
    my_config["model_args"],
    lr=4e-3,
    save_dir=my_dir,
    num_training_points=X_train.shape[0],
    num_stochastic_modules=1,
    beta_elbo=1.0,
    num_mc_samples_train=10,
    num_mc_samples_test=50,
    output_noise_scale=1.3,
    prior_mu=0.0,
    prior_sigma=1.0,
    posterior_mu_init=0.0,
    posterior_rho_init=-3.0,
    alpha=1.0,
)

logger = CSVLogger(my_dir)

pl_args = {
    "max_epochs": max_epochs,
    "logger": logger,
    "log_every_n_steps": 1,
    "accelerator": "gpu",
    "devices": [5],
    "limit_val_batches": 0.0,
}
trainer = Trainer(**pl_args)

# fit model
trainer.fit(base_model, dm)

# using the trainer does save predictions
trainer.test(base_model, dm.test_dataloader())
csv_path = os.path.join(my_dir, "predictions.csv")

pred = base_model.predict_step(X_test)

my_fig = plot_predictions(
    X_train,
    y_train,
    X_test,
    y_test,
    pred["mean"],
    pred["pred_uct"],
    epistemic=pred.get("epistemic_uct", None),
    aleatoric=pred.get("aleatoric_uct", None),
    title="BNN with VI",
)
plt.savefig("preds.png")
plt.show()

metrics_path = os.path.join(my_dir, "lightning_logs", "version_0", "metrics.csv")
df = pd.read_csv(metrics_path)

train_loss = df[df["train_loss"].notna()]["train_loss"]
train_rmse = df[df["train_RMSE"].notna()]["train_RMSE"]

fig, ax = plt.subplots(2)
ax[0].plot(np.arange(len(train_loss)), train_loss)
ax[0].set_title("Train Loss")

ax[1].plot(np.arange(len(train_rmse)), train_rmse)
ax[1].set_title("Train RMSE")

plt.show()
plt.savefig("loss_curves.png")
# import pdb

# pdb.set_trace()

# print(0)