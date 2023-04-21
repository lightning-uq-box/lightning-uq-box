"""Multi-SWAG."""

from typing import Any, Dict, List, Type, Union

import numpy as np
import torch
from torch import Tensor

from uq_method_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .deep_ensemble_model import DeepEnsembleModel
from .swag import SWAGModel


class MultiSWAG(DeepEnsembleModel):
    """Multi-SWAG."""

    def __init__(
        self,
        ensemble_members: List[Dict[str, Union[Type[SWAGModel], str]]],
        num_mc_samples: int,
        save_dir: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Multi-SWAG Wrapper.

        Args:
            swag_members: List of dicts where each element specifies the
                SWAG class and a path to a checkpoint
            num_mc_samples: number of MC samples to draw per single swag member
            save_dir: path to directory where to store prediction
            quantiles: quantile values to compute for prediction
        """
        super().__init__(ensemble_members, save_dir, quantiles)

        self.hparams["num_mc_samples"] = num_mc_samples
        self.hparams["quantiles"] = quantiles

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        """Conduct forward pass for inference with Multi-SWAG model.

        Args:
            X: input Tensor

        Returns:
            stacked tensor of dim [batch_size, num_outputs, n_swag_models*n_samples]

        """
        out: List[torch.Tensor] = []
        for model_config in self.hparams.ensemble_members:
            loaded_swag_model: SWAGModel = model_config[
                "model_class"
            ].load_from_checkpoint(
                model_config["ckpt_path"],
                model=model_config["base_model"],
                train_loader=model_config["train_loader"],
            )
            for i in range(self.hparams.num_mc_samples):
                loaded_swag_model.sample_state()
                out.append(loaded_swag_model(X))
        return torch.stack(out, dim=-1)

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Compute prediction step for a deep ensemble.

        Args:
            X: input tensor of shape [batch_size, input_di]

        Returns:
            prediction dictionary
        """
        with torch.no_grad():
            preds = self.forward(X).cpu().numpy()

        mean_samples = preds[:, 0, :]

        if preds.shape[1] == 2:
            log_sigma_2_samples = preds[:, 1, :]
            eps = np.ones_like(log_sigma_2_samples) * 1e-6
            sigma_samples = np.sqrt(eps + np.exp(log_sigma_2_samples))
            mean = mean_samples.mean(-1)
            std = compute_predictive_uncertainty(mean_samples, sigma_samples)
            aleatoric = compute_aleatoric_uncertainty(sigma_samples)
            epistemic = compute_epistemic_uncertainty(mean_samples)
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
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
            quantiles = compute_quantiles_from_std(mean, std, self.hparams.quantiles)
            return {
                "mean": mean,
                "pred_uct": std,
                "epistemic_uct": std,
                "lower_quant": quantiles[:, 0],
                "upper_quant": quantiles[:, -1],
            }
