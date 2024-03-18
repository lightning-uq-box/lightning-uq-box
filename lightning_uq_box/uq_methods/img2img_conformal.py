# Adapted from https://github.com/aangelopoulos/im2im-uq?tab=readme-ov-file
# MIT License
# Copyright (c) 2021 Anastasios Angelopoulos

# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"Image-to-Image Conformal Uncertainty Estimation"

import torch
import numpy as np
from typing import Optional, Any
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

from .base import PosthocBase
from .utils import freeze_model_backbone, default_regression_metrics


class Img2ImgConformal(PosthocBase):
    """Image-to-Image Conformal Uncertainty Estimation

    This module is a wrapper around a base model that provides
    conformal uncertainty estimates for image-to-image tasks.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2202.05265
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.9,
        delta: float = 0.1,
        min_lambda: float = 0.01,
        max_lambda: float = 1.0,
        num_lambdas: int = 100,
        freeze_backbone: bool = False,
    ):
        """Initialize a new Img2ImgConformal instance.

        Args:
            base_model: the base model to wrap
            alpha: the significance level for the conformal prediction
        """
        super().__init__(model)
        self.delta = delta
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.num_lambdas = num_lambdas

        self.model = model
        self.alpha = alpha
        self.freeze_backbone = freeze_backbone
        self.freeze_model()

    def freeze_model(self) -> None:
        """Freeze model backbone.

        By default, assumes a timm model with a backbone and head.
        Alternatively, selected the last layer with parameters to freeze.
        """
        if self.freeze_backbone:
            freeze_model_backbone(self.model)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def forward(self, X: Tensor, lam: Optional[float]) -> dict[str, Tensor]:
        """Forward pass of model.

        Args:
            X: input tensor of shape [batch_size x input_dims]
            lam: The lambda parameter. Default is None.

        Returns:
            model output tensor of shape [batch_size x num_outputs]
        """
        with torch.no_grad():
            if hasattr(self.model, "predict_step"):
                pred = self.model.predict_step(X)
                pred = torch.stack([pred["lower"], pred["pred"], pred["upper"]], dim=1)
            else:
                pred = self.model(X)

        # conformalize in this step
        pred = self.adjust_model_logits(pred, lam)

        return pred

    def adjust_model_logits(
        self, output: Tensor, lam: Optional[float] = None
    ) -> tuple:
        """
        Compute the nested sets from the output of the model.

        Args:
            output: The output tensor.
            lam: The lambda parameter. Default is None.

        Returns:
            The lower edge, the output, and the upper edge.
        """
        if lam is None:
            lam = self.lam
        output[:, 0, :, :, :] = torch.minimum(
            output[:, 0, :, :, :], output[:, 1, :, :, :] - 1e-6
        )
        output[:, 2, :, :, :] = torch.maximum(
            output[:, 2, :, :, :], output[:, 1, :, :, :] + 1e-6
        )
        upper_edge = (
            lam * (output[:, 2, :, :, :] - output[:, 1, :, :, :])
            + output[:, 1, :, :, :]
        )
        lower_edge = output[:, 1, :, :, :] - lam * (
            output[:, 1, :, :, :] - output[:, 0, :, :, :]
        )

        return {
            "lower": lower_edge,
            "pred": output[:, 1, :, :, :],
            "upper": upper_edge,
        }
    
    def rcps_loss_fn(self, pset, label):
        """RCPS Loss function, fraction_missed_loss by default"""
        misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
        misses[misses > 1.0] = 1.0
        d = len(misses.shape)
        return misses.mean(dim=tuple(range(1,d)))
    

    def get_rcps_losses_from_outputs(self, logit_dataset: Dataset, rcps_loss_fn, lam: float):
        """Compute the RCPS loss from the model outputs.
        
        Args:
            logit_dataset: The logit dataset.
            rcps_loss_fn: The RCPS loss function.
            lam: The lambda parameter
        """
        losses = []
        dataloader = DataLoader(logit_dataset, batch_size=self.trainer.datamodule.batch_size, shuffle=False, num_workers=0, pin_memory=True) 
        for batch in dataloader:
            x, labels = batch
            x = x.to(self.trainer.device)
            labels = labels.to(self.trainer.device)
            sets = self.nested_sets_from_output(x,lam) 
            losses = losses + [rcps_loss_fn(sets, labels)]
        return torch.cat(losses,dim=0)


    def on_validation_epoch_end(self) -> None:
        """Perform Img2Img calibration

        Args:
            outputs: list of dictionaries containing model outputs and labels
        """
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()
    
        # probably store all of this in cpu memory instead
        out_dataset = TensorDataset(all_logits,all_labels)

        lambdas = torch.linspace(self.min_lambda, self.max_lambda, self.num_lambdas)

        self.calib_loss_table = torch.zeros((all_labels.shape[0], lambdas.shape[0]))

        dlambda = lambdas[1] - lambdas[0]
        self.lam = lambdas[-1]+dlambda-1e-9
        for lam in reversed(lambdas):

            losses = self.get_rcps_losses_from_ouptuts(out_dataset, lam-dlambda)
            self.calib_loss_table[:,np.where(lambdas==lam)[0]] = losses[:,None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], self.delta)
            if Rhat >= self.alpha or RhatPlus > self.alpha:
                self.lam = lam
                break

        self.post_hoc_fitted = True

    def test_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Tensor]:
        """Test step.
        
        Args:
            batch: batch of testing data
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        pred_dict = self.predict_step(batch[self.input_key])

        return pred_dict

    def predict_step(self, X: Tensor) -> dict[str, Tensor]:
        """Prediction step with applied temperature scaling.

        Args:
            X: input tensor of shape [batch_size x num_features]
        """
        if not self.post_hoc_fitted:
            raise RuntimeError(
                "Model has not been post hoc fitted, "
                "please call trainer.validate(model, datamodule) first."
            )
        
        return self.forward(X)
        


def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.minimum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

### UCB of mean via Hoeffding-Bentkus hybridization
def HB_mu_plus(muhat, n, delta, maxiters=1000):
    """"""
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        try:
          return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)
        except:
          print(f"BRENTQ RUNTIME ERROR at muhat={muhat}")
          return 1.0