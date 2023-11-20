"""Regularized Adaptive Prediction Sets (RAPS)."""

"""Adapted from https://github.com/aangelopoulos/conformal_classification"""

from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from .base import PosthocBase
from .temp_scaling import run_temperature_optimization, temp_scale_logits
from .utils import _get_num_inputs, _get_num_outputs, default_classification_metrics


class RAPS(PosthocBase):
    """Regularized Adaptive Prediction Sets (RAPS).

    If you use this method, please cite the following paper:

    * https://arxiv.org/abs/2009.14193
    """

    def __init__(
        self,
        model: Union[LightningModule, nn.Module],
        optim_lr: float = 0.01,
        max_iter: int = 50,
        alpha: float = 0.1,
        kreg: float = None,
        lambda_param: float = None,
        randomized: bool = False,
        allow_zero_sets: bool = False,
        pct_param_tune: float = 0.3,
        lambda_criterion: str = "size",
    ) -> None:
        """Initialize RAPS.

        Args:
            model: model to be calibrated with Temperature S
            optim_lr: learning rate for optimizer
            max_iter: maximum number of iterations to run optimizer
        """
        super().__init__(model)

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optim_lr = optim_lr
        self.max_iter = max_iter
        self.criterion = nn.CrossEntropyLoss()

        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets
        self.pct_param_tune = pct_param_tune
        self.num_classes = _get_num_outputs(model)

        self.kreg = kreg
        self.lambda_param = lambda_param
        if kreg == None or lambda_param == None:
            kreg, lamda, calib_logits = pick_parameters(
                model,
                calib_logits,
                alpha,
                kreg,
                lamda,
                randomized,
                allow_zero_sets,
                pct_param_tune,
                batch_size,
                lamda_criterion,
            )

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Test step after running posthoc fitting methodology."""
        raise NotImplementedError

    def adjust_model_logits(self, model_logits: Tensor) -> Tensor:
        """Adjust model output according to post-hoc fitting procedure.

        Args:
            model_logits: model output tensor of shape [batch_size x num_outputs]

        Returns:
            adjusted model logits tensor of shape [batch_size x num_outputs]
        """
        raise NotImplementedError

    def on_validation_end(self) -> None:
        """Apply RASP conformal method."""
        all_logits = torch.cat(self.model_logits, dim=0).detach()
        all_labels = torch.cat(self.labels, dim=0).detach()

        optimizer = partial(torch.optim.LBFGS, lr=self.optim_lr, max_iter=self.max_iter)
        self.temperature = run_temperature_optimization(
            optimizer, self.temperature, all_logits, all_labels, self.criterion
        )
        import pdb

        pdb.set_trace()
        print(0)


# def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
#     num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
#     paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
#     calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
#     paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

#     if kreg == None:
#         kreg = pick_kreg(paramtune_logits, alpha)
#     if lamda == None:
#         if lamda_criterion == "size":
#             lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
#         elif lamda_criterion == "adaptiveness":
#             lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
#     return kreg, lamda, calib_logits


# def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
#     # Calculate lamda_star
#     best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes
#     # Use the paramtune data to pick lamda.  Does not violate exchangeability.
#     for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]: # predefined grid, change if more precision desired.
#         conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
#         top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
#         if sz_avg < best_size:
#             best_size = sz_avg
#             lamda_star = temp_lam
#     return lamda_star

# def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
#     # Calculate lamda_star
#     lamda_star = 0
#     best_violation = 1
#     # Use the paramtune data to pick lamda.  Does not violate exchangeability.
#     for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]: # predefined grid, change if more precision desired.
#         conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
#         curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
#         if curr_violation < best_violation:
#             best_violation = curr_violation
#             lamda_star = temp_lam
#     return lamda_star
