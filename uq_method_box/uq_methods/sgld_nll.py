"""Stochastic Gradient Langevin Dynamics (SGLD) model."""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from torch.utils.data import DataLoader

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std
)

from uq_method_box.uq_methods import BaseModel
from uq_method_box.train_utils import NLL



class SGLD(Optimizer):
    def __init__(self, params, lr=required, noise_factor=1.0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, noise_factor=noise_factor, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            noise_factor = group['noise_factor']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)
                p.data.add_(noise_factor * (2.0 * group['lr']) ** 0.5, torch.randn_like(d_p))

        return loss
    

class SGLDModel(BaseModel):
    """SGLD method for regression."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module = None,
        criterion: nn.Module =  NLL(),
    ) -> None:
        
        super().__init__(config, model, criterion)
        
        self.n_burnin_epochs = self.config["model"]["n_burnin_epochs"]
        self.n_sgld_samples = self.config["model"]["n_sgld_samples"]
        self.max_epochs=self.config["pl"]["max_epochs"]  
        self.models: List[nn.Module] = []
        self.quantiles = self.config["model"]["quantiles"]

        assert( self.n_sgld_samples + self.n_burnin_epochs <= self.config["pl"]["max_epochs"]), "The number of max epochs needs to be larger than the sum of the burnin phase and model gathering"

        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers, with SGLD optimizer.
        """
        optimizer = SGLD(
            self.model.parameters(), lr=self.config["optimizer"]["lr"]
        )
        return {"optimizer": optimizer}

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss and List of models 
        """

    

        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        if self.current_epoch > self.n_burnin_epochs: 
            self.models.append(copy.deepcopy(self.model))

        return loss, self.models


    def extract_mean_output(self, out: Tensor) -> Tensor:
        """Extract the mean output from model prediction.

        Args:
            out: output from :meth:`self.forward` [batch_size x (mu, sigma)]

        Returns:
            extracted mean used for metric computation [batch_size x 1]
        """
        assert (
            out.shape[-1] == 2
        ), "This model should give exactly 2 outputs due to NLL loss estimation."
        return out[:, 0:1]
    
    
   
    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Predict step with SGLD, take n_sgld_sampled models in list of models obtained after n_burnin_epochs and corresponding learning rates (initially constant), get mean and variance like for ensembles.
        Args: 
            self: SGLD class
            batch_idx: default int=0
            dataloader_idx: default int=0

        Returns:
            "mean": sgld_mean, mean prediction over models
            "pred_uct": sgld_std, predictive uncertainty
            "epistemic_uct": sgld_epistemic, epistemic uncertainty
            "aleatoric_uct": sgld_aleatoric, aleatoric uncertainty, averaged over models
            "quantiles": sgld_quantiles, quantiles assuming output is Gaussian
        """
        

        preds = (
            torch.stack([self.model(batch) for self.model in self.models], dim=-1)
            .detach()
            .numpy()
        )  # shape [n_sgld_samples, batch_size, num_outputs]
        
        # Prediction gives two outputs, due to NLL loss
        mean_samples = preds[:, 0, :]
        sigma_samples = preds[:, 1, :]

        sgld_mean = mean_samples.mean(-1)
        sgld_std = compute_predictive_uncertainty(mean_samples, sigma_samples)
        sgld_aleatoric = compute_aleatoric_uncertainty(sigma_samples)
        sgld_epistemic = compute_epistemic_uncertainty(mean_samples)

        #now compute quantiles, dimension is len(self.quantiles)

        sgld_quantiles = compute_quantiles_from_std(sgld_mean, sgld_std, self.quantiles)

        return {
            "mean": sgld_mean,
            "pred_uct": sgld_std,
            "epistemic_uct": sgld_epistemic,
            "aleatoric_uct": sgld_aleatoric,
            "quantiles": sgld_quantiles
        }
    
    