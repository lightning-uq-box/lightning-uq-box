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
#
# To demonstrate the method, we will make use of a Toy Regression Example that is defined as a [Lightning Datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). While this might seem like overkill for a small toy problem, we think it is more helpful how the individual pieces of the library fit together so you can train models on more complex tasks.


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
#
# For our Toy Regression problem, we will use a simple Multi-layer Perceptron (MLP) that will be converted to a BNN as well as a Latent Variable Network.


bnn = MLP(
    n_inputs=1, n_hidden=[25, 25, 25, 25, 25], n_outputs=1, activation_fn=nn.ReLU()
)
latent_net = MLP(
    n_inputs=2,  # num_input_features + num_target_dim
    n_outputs=2,  # 2 * lv_latent_dim
    n_hidden=[20],
    activation_fn=nn.ReLU(),
)
bnn, latent_net


# With an underlying neural network, we can now use our desired UQ-Method as a sort of wrapper. All UQ-Methods are implemented as [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) that allow us to concisely organize the code and remove as much boilerplate code as possible.
#
# The BNN_LV_VI_Batched_Regression model enables us to chose the layers of our network we want to make stochastic via the ```stochastic_module_names``` argument. This can be done by either a list of module names or a list of module numbers. For example, stochastic_module_name = [-1] would only make the last layer stochastic while all other layers remain determinstic.
# The default value of None makes all layers stochastic. The hyperparameter ```alpha``` determines how the Gaussian approximation of the posterior fits the posterior of the weights, see Figure 1, [Depeweg, 2016](https://arxiv.org/abs/1605.07127).


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
#
# Now that we have a LightningDataModule and a UQ-Method as a LightningModule, we can conduct training with a [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html). It has tons of options to make your life easier, so we encourage you to check the documentation.


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
# To get some insights into how the training went, we can use the utility function to plot the training loss and RMSE metric.


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


# For some additional metrics relevant to UQ, we can use the great [uncertainty-toolbox](https://uncertainty-toolbox.github.io/) that gives us some insight into the calibration of our prediction.


fig = plot_calibration_uq_toolbox(
    preds["pred"].numpy(),
    preds["pred_uct"].numpy(),
    y_test.cpu().numpy(),
    X_test.cpu().numpy(),
)
