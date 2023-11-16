import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (
    NLL,
    BNN_VI_ELBO_Classification,
    BNN_VI_ELBO_Regression,
)
from lightning_uq_box.viz_utils import (
    plot_predictions_classification,
    plot_predictions_regression,
    plot_training_metrics,
    plot_two_moons_data,
)

plt.rcParams["figure.figsize"] = [14, 5]


seed_everything(1)  # seed everything for reproducibility

my_temp_dir = tempfile.mkdtemp()

dm = ToyHeteroscedasticDatamodule(batch_size=50)

X_train, y_train, X_test, y_test = (dm.X_train, dm.y_train, dm.X_test, dm.y_test)

network = MLP(n_inputs=1, n_hidden=[50, 50], n_outputs=2, activation_fn=nn.Tanh())

bbp_model = BNN_VI_ELBO_Regression(
    network,
    optimizer=partial(torch.optim.Adam, lr=1e-2),
    criterion=NLL(),
    part_stoch_module_names=[-1],
    num_training_points=X_train.shape[0],
    num_mc_samples_train=10,
    num_mc_samples_test=25,
    burnin_epochs=20,
)

logger = CSVLogger(my_temp_dir)
trainer = Trainer(
    max_epochs=50,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=20,
    enable_checkpointing=False,
    enable_progress_bar=False,
)

trainer.fit(bbp_model, dm)

preds = bbp_model.predict_step(X_test)

# import pdb

# pdb.set_trace()

fig = plot_training_metrics(my_temp_dir, "RMSE")

fig = plot_predictions_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    preds["pred"].squeeze(-1),
    preds["pred_uct"],
    epistemic=preds["epistemic_uct"],
    aleatoric=preds["aleatoric_uct"],
    title="Bayes By Backprop MFVI",
    show_bands=False,
)


plt.show()
