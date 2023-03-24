import os
import tempfile
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch import seed_everything

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule, ToySineDatamodule
from uq_method_box.eval_utils import compute_aleatoric_uncertainty
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BaseModel, DeterministicGaussianModel
from uq_method_box.viz_utils import plot_predictions

seed_everything(4)


dm = ToyHeteroscedasticDatamodule()

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

my_config = {
    "model_args": {"n_inputs": 1, "n_outputs": 2},
    "loss_fn": "nll",
    "swag_args": {"num_mc_samples": 100, "swag_lr": 1e-3, "loss_fn": "nll"},
}

my_dir = tempfile.mkdtemp()

max_epochs = 700

base_model = DeterministicGaussianModel(
    MLP,
    my_config["model_args"],
    optimizer=torch.optim.Adam,
    optimizer_args={"lr": 0.03},
    loss_fn=my_config["loss_fn"],
    save_dir=my_dir,
    burnin_epochs=20,
    max_epochs=max_epochs,
)

trainer = Trainer(max_epochs=max_epochs, logger=False)

# fit model
trainer.fit(base_model, dm)

# using the trainer does save predictions
trainer.test(base_model, dm.test_dataloader())
csv_path = os.path.join(my_dir, "predictions.csv")

from uq_method_box.uq_methods import SWAGModel

# fit laplace
swag_model = SWAGModel(
    base_model,
    num_swag_epochs=50,
    max_swag_snapshots=50,
    num_mc_samples=my_config["swag_args"]["num_mc_samples"],
    snapshot_freq=1,
    swag_lr=my_config["swag_args"]["swag_lr"],
    loss_fn=my_config["loss_fn"],
    train_loader=dm.train_dataloader(),
    save_dir=my_dir,
)

pred = swag_model.predict_step(X_test)

swag_fig = plot_predictions(
    X_train,
    y_train,
    X_test,
    y_test,
    pred["mean"],
    pred["pred_uct"],
    epistemic=pred["epistemic_uct"],
    aleatoric=pred.get("aleatoric_uct", None),
    title="SWAG",
)


import pdb

pdb.set_trace()

print(0)
