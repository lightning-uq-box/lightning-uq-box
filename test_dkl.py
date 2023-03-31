"""Test DKL implementation."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.grid import choose_grid_size
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule

# from uq_method_box.eval_utils import compute_aleatoric_uncertainty
from uq_method_box.models import MLP
from uq_method_box.uq_methods import (  # BaseModel,; DeterministicGaussianModel,
    DeepKernelLearningModel,
    DKLGPLayer,
    initial_values,
)
from uq_method_box.viz_utils import plot_predictions

seed_everything(4)

# define datamodule

dm = ToyHeteroscedasticDatamodule()

# define data
X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

# config args
my_config = {
    "model_args": {"n_inputs": 1, "n_outputs": 1, "n_hidden": [100]},
    "loss_fn": "mse",
    "gp_args": {"n_outputs": 1},
}


# temporary directory for saving
my_dir = tempfile.mkdtemp()

# define untrained feature extractor
feature_extractor = MLP(**my_config["model_args"])

# remove output layer to just have feature extractor
# feature_extractor.model[-1] = torch.nn.Identity()
grid_size = choose_grid_size(train_inputs=X_train, ratio=1.0, kronecker_structure=True)

# get number of inducing points
n_inducing_points = 100


# get intial inducing points and initial lengthscale for kernels
initial_inducing_points, initial_lengthscale = initial_values(
    train_loader.dataset, feature_extractor, n_inducing_points
)

# define dict for gp args from initial computations above
gp_args = {
    "n_outputs": my_config["gp_args"]["n_outputs"],
    "grid_size": grid_size,
    "num_dims": my_config["model_args"]["n_outputs"],
    "kiss": False,  # use KISS approximation
    "initial_lengthscale": initial_lengthscale,
    "initial_inducing_points": initial_inducing_points,
    "kernel": "RQ",
}

# import pdb
# pdb.set_trace()
# gp_args["initial_lengthscale"] = 1


# define dkl model

dkl_model = DeepKernelLearningModel(
    feature_extractor,
    gp=DKLGPLayer,
    gp_args=gp_args,
    elbo_fn=VariationalELBO,
    n_gp_samples=30,
    n_train_points=X_train.shape[0],
    lr=1e-2,
    save_dir=my_dir,
)


max_epochs = 400

logger = CSVLogger(my_dir)
trainer = Trainer(max_epochs=max_epochs, logger=logger, log_every_n_steps=1)

# fit model
trainer.fit(dkl_model, dm)

# using the trainer does save predictions
trainer.test(dkl_model, dm.test_dataloader())
csv_path = os.path.join(my_dir, "predictions.csv")

with torch.no_grad():
    pred = dkl_model.predict_step(X_test)

dkl_fig = plot_predictions(
    X_train,
    y_train,
    X_test,
    y_test,
    pred["mean"],
    pred["pred_uct"],
    epistemic=pred["epistemic_uct"],
    aleatoric=pred.get("aleatoric_uct", None),
    title="DKL",
)

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
# import pdb

# pdb.set_trace()

# print(0)
