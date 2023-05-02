"""BNN+VI Testing Script Toy Data."""

import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from uq_method_box.datamodules import ToyBimodalDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BNN_LV_VI
from uq_method_box.viz_utils import plot_predictions

# seed_everything(4)
torch.set_float32_matmul_precision("medium")

dm = ToyBimodalDatamodule(batch_size=100)

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X,
    dm.y,
    dm.train_dataloader(),
    dm.X,
    dm.y,
    dm.test_dataloader(),
)

my_config = {
    "model_args": {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_hidden": [20, 20],
        "activation_fn": torch.nn.ReLU(),
    },
    "loss_fn": "nll",
    "latent_net": MLP(
        n_inputs=2,  # num_input_features + num_target_dim
        n_outputs=2,  # 2 * lv_latent_dim
        n_hidden=[20],
        activation_fn=torch.nn.ReLU(),
    ),
}

my_dir = tempfile.mkdtemp()

max_epochs = 1000

base_model = BNN_LV_VI(
    MLP,
    my_config["model_args"],
    lr=1e-3,
    save_dir=my_dir,
    num_training_points=X_train.shape[0],
    num_stochastic_modules=3,
    beta_elbo=1.0,
    num_mc_samples_train=10,
    num_mc_samples_test=50,
    output_noise_scale=1.3,
    prior_mu=0.0,
    prior_sigma=1.0,
    posterior_mu_init=0.0,
    posterior_rho_init=-3.0,
    alpha=1.0,
    latent_net=my_config["latent_net"],
)

logger = CSVLogger(my_dir)

# for cpu training set 'accelerator' flag to 'cpu' and comment out the devices flag
pl_args = {
    "max_epochs": max_epochs,
    "logger": logger,
    "log_every_n_steps": 1,
    "accelerator": "cpu",
    # "devices": [0],
    "limit_val_batches": 0.0,
}
trainer = Trainer(**pl_args)

# fit model
start = time.time()
trainer.fit(base_model, dm)
print(f"Fit took {time.time() - start} seconds.")

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
