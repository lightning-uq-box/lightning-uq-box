"""Base Model for UQ methods."""

import os
from typing import Any, Dict, List

import pyro
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .utils import save_predictions_to_csv


class BaseModel(LightningModule):
    """Deterministic Base Trainer as LightningModule."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module = None,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """Initialize a new Base Model."""
        super().__init__()
        self.config = config

        if model is not None:
            self.model = model
        else:
            self.model = timm.create_model(
                config["model"]["model_name"],
                pretrained=True,
                in_chans=self.config["model"]["in_chans"],
                num_classes=self.config["model"]["num_outputs"],
            )

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="test_",
        )

        self.criterion = criterion

        self.save_hyperparameters(
            ignore=["criterion", "train_metrics", "val_metrics", "test_metrics"]
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Different models have different number of outputs, i.e. Gaussian NLL 2
        or quantile regression but for the torchmetrics only
        the mean/median is considered.

        Args:
            out: output from :meth:`self.forward` [batch_size x num_outputs]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert out.shape[-1] <= 2, "Ony support single mean or Gaussian output."
        return out[:, 0:1]

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss

    def training_epoch_end(self, outputs: Any) -> None:
        """Log epoch-level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch

        Returns:
            validation loss
        """
        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("val_loss", loss)  # logging to Logger
        self.val_metrics(self.extract_mean_output(out), y)

        return loss

    def validation_epoch_end(self, outputs: Any) -> None:
        """Log epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Test step is in most cases unique to the different methods."""
        raise NotImplementedError

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Log epoch level validation metrics.

        Args:
            outputs: list of items returned by test step, dictionaries
        """
        save_predictions_to_csv(
            outputs,
            os.path.join(self.config["experiment"]["save_dir"], "predictions.csv"),
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config["optimizer"]["lr"]
        )
        return {"optimizer": optimizer}


class EnsembleModel(LightningModule):
    """Base Class for different Ensemble Models."""

    def __init__(
        self, config: Dict[str, Any], ensemble_members: List[nn.Module]
    ) -> None:
        """Initialize a new instance of DeepEnsembleModel Wrapper.

        Args:
            config: configuration dictionary
            ensemble_members: instantiated ensemble members
        """
        super().__init__()
        self.config = config

        self.ensemble_members = ensemble_members

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward step of Deep Ensemble.

        Args:
            batch:

        Returns:
            Ensemble member outputs stacked over last dimension for output
            of [batch_size, num_outputs, num_ensemble_members]
        """
        raise NotImplementedError

    def test_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Any:
        """Compute test step for deep ensemble and log test metrics.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            dictionary of uncertainty outputs
        """
        target = batch[1]
        out_dict = self.predict_step(batch)
        out_dict["targets"] = target.detach().squeeze(-1).numpy()
        return out_dict

    def test_epoch_end(self, outputs: Any) -> None:
        """Log epoch level validation metrics.

        Args:
            outputs: list of items returned by test step, dictionaries
        """
        save_predictions_to_csv(
            outputs,
            os.path.join(self.config["experiment"]["save_dir"], "predictions.csv"),
        )

    def generate_ensemble_predictions(self, batch: Any) -> Tensor:
        """Generate ensemble predictions.

        Args:
            batch: data batch

        Returns:
            ensemble predictions of shape [batch_size, num_outputs, num_ensemble_preds]
        """
        raise NotImplementedError

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Any:
        """Compute prediction step for a deep ensemble.

        Args:
            batch: prediction batch of shape [batch_size x input_dims]

        Returns:
            mean and standard deviation of MC predictions
        """
        preds = self.generate_ensemble_predictions(batch)

        mean_samples = preds[:, 0, :].detach().numpy()

        # assume nll prediction with sigma
        if preds.shape[1] == 2:
            sigma_samples = preds[:, 1, :].detach().numpy()
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            quantiles = compute_quantiles_from_std(
                mean, std, self.config["model"]["quantiles"]
            )
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": epistemic,
                "aleatoric_uct": aleatoric,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
        # assume mse prediction
        else:
            mean = mean_samples.mean(-1)
            std = mean_samples.std(-1)
            quantiles = compute_quantiles_from_std(
                mean, std, self.config["model"]["quantiles"]
            )

            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": std,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }


# PYRO BNN Base
class PyroOptWrap(pyro.infer.SVI):
    """Wrapper for Pytorch Lightning to give a state_dict."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a new wrapper."""
        super().__init__(*args, **kwargs)

    def state_dict(self):
        """Return dummy state dict."""
        return {}


class PyroBNNBase(BaseModel):
    """Bayesian Neural Network with Latent Variables."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize a new instance of BNN_LV.

        Args:
            config:
        """
        super().__init__(config, None, None)  # no criterion

        # how to save checkpoints from this model not sure
        # https://pyro4ci.readthedocs.io/en/latest/_modules/pyro/params/param_store.html

    def model(self, batch):
        """Define the Pyro Model https://pyro.ai/examples/svi_part_i.html.

        Args:
            batch: input batch of data
        """
        raise NotImplementedError

    def guide(self, batch):
        """Define the Pyro Guide https://pyro.ai/examples/svi_part_i.html.

        Args:
            batch: input batch of data
        """
        raise NotImplementedError

    def forward(self, x):
        """Define the forward pass through the model.

        The forward function should be called in the model function.

        Args:
            x: input data x
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Compute training Step for BNN+LV with SVI.

        Args:
            batch:
            batch_idx:
        """
        loss = self.svi.step(batch)
        loss = torch.tensor(loss).requires_grad_(True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation Step for BNN+LV with SVI.

        Args:
            batch:
            batch_idx:
        """
        loss = self.svi.evaluate_loss(batch)
        return loss

    def configure_optimizers(self):
        """Configure Pyro Optimizer."""
        self.svi = PyroOptWrap(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam(
                {"lr": self.config["optimizer"]["lr"], "momentum": 0.0}
            ),
            loss=pyro.infer.Trace_ELBO(),
        )

        return [self.svi]
