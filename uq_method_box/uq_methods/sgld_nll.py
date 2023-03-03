"""Stochastic Gradient Langevin Dynamics (SGLD) model."""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Any, Dict, List

from uq_method_box.uq_methods import BaseModel
from uq_method_box.train_utils import NLL

#TO DOs:
# change optimizer for training to make gradient updates: 
"""Implementation of SGLD algorithm as described in [1].  The loss is assumed to be the average minibatch loss, hence
  of the form
      -(1/m) \sum_{i=1}^m log p(y_i|x_i,w) - (1/n) log p(w),
  where m is the batch size and n is the total number of iid training samples.
  #### References
  [1] _Max Welling_ and _Yee Whye Teh_, NOTYPO
      "bayesian Learning via Stochastic Gradient Langevin Dynamics",
      ICML 2011,
      [PDF](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)  """


"""SGLD optimizer class from: https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html """
# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md

class SGLD(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=120,
                 diagonal_bias=1e-8) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)



    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                    1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                    0.5 * preconditioner * gradient * num_pseudo_batches +
                    torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss
    

class SGLDModel(BaseModel):
    """SGLD method for regression."""

    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module =  NLL(),
    ) -> None:
        super().__init__(config, model, criterion)
        
        self.n_burnin_pochs = self.config["model"]["base_model"]["n_burnin_pochs"]
        self.n_sgld_samples = self.config["model"]["base_model"]["n_sgld_samples"]
        self.max_poch=self.config["pl"]["max_epoch"] 
        #do I need to define this? 
        self.Models = List[nn.Module]

        
    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = SGLD(self.model,
            self.model.parameters(), lr=self.config["optimizer"]["lr"]
        )
        return {"optimizer": optimizer}

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss and Models (can I do this?)
        """
        #need something to interrupt training step after n_burnin_epochs
        #need to check if self.config["pl"]["max_epoch"] >= n_sgld_samples + n_burnin_epochs
        #the another n_sgld_samples runs and save weights for ensemble
     
        if self.config["pl"]["max_epoch"] >= self.n_sgld_samples + self.n_burnin_epochs:
            #no error, continue
        else:
            #send error message/print error
            #return print("Too little epochs ")
            
        #does this work?

        if self.current_epoch > self.n_burnin_epochs: 
            self.Models.append(copy.deepcopy(self.model))

        X, y = args[0]
        out = self.forward(X)
        loss = self.criterion(out, y)

        self.log("train_loss", loss)  # logging to Logger
        self.train_metrics(self.extract_mean_output(out), y)

        return loss, self.Models

    def training_epoch_end(self, outputs: Any) -> None:
        """Log epoch-level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()


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
        """Predict step with SGLD ."""
        #take n_sgld_sampled models in List Models obtained after n_burnin_epochs and corresponding learning rates (initially constant), get mean and variance
        
                #pseudo code
        preds = (
            torch.stack([self.model(batch) for self.model in self.Models], dim=-1)
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

        return {
            "mean": sgld_mean,
            "pred_uct": sgld_std,
            "epistemic_uct": sgld_epistemic,
            "aleatoric_uct": sgld_aleatoric,
        }
    
    