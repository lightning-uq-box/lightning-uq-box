import os
import sys
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (
    BNN_LV_VI_Batched_Regression,
    BNN_LV_VI_Regression,
)
from lightning_uq_box.viz_utils import (
    plot_calibration_uq_toolbox,
    plot_predictions_regression,
    plot_toy_regression_data,
    plot_training_metrics,
)

plt.rcParams["figure.figsize"] = [14, 5]


seed_everything(0)  # seed everything for reproducibility


# We define a temporary directory to look at some training metrics and results.


my_temp_dir = tempfile.mkdtemp()


# ## Datamodule


dm = ToyHeteroscedasticDatamodule(batch_size=32)

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)


fig = plot_toy_regression_data(X_train, y_train, X_test, y_test)


# ## Model


bnn = MLP(
    n_inputs=1, n_hidden=[25, 25, 25, 25, 25], n_outputs=1, activation_fn=nn.ReLU()
)
latent_net = MLP(
    n_inputs=2,  # num_input_features + num_target_dim
    n_outputs=2,  # 2 * lv_latent_dim
    n_hidden=[20],
    activation_fn=nn.ReLU(),
)


bnn_vi_model = BNN_LV_VI_Batched_Regression(
    bnn,
    latent_net,
    optimizer=partial(torch.optim.Adam, lr=1e-3),
    num_training_points=X_train.shape[0],
    n_mc_samples_train=10,
    n_mc_samples_test=50,
    output_noise_scale=1.3,
    prior_mu=0.0,
    prior_sigma=1.0,
    posterior_mu_init=0.0,
    posterior_rho_init=-6.0,
    alpha=1e-6,
    bayesian_layer_type="reparameterization",
)


# ## Trainer


logger = CSVLogger(my_temp_dir)
trainer = Trainer(
    max_epochs=250,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=1,
    enable_checkpointing=False,
    enable_progress_bar=True,
    limit_val_batches=0.0,  # skip validation runs
    default_root_dir=my_temp_dir,
)


# Training our model is now easy:


trainer.fit(bnn_vi_model, dm)


# ## Training Metrics
#


fig = plot_training_metrics(
    os.path.join(my_temp_dir, "lightning_logs"), ["train_loss", "trainRMSE"]
)


# ## Prediction


# save predictions
trainer.test(bnn_vi_model, dm)


preds = bnn_vi_model.predict_step(X_test)


# ## Evaluate Predictions


fig = plot_predictions_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    preds["pred"],
    preds["pred_uct"],
    epistemic=preds["epistemic_uct"],
    aleatoric=preds["aleatoric_uct"],
    title="title here",
)

plt.show()
