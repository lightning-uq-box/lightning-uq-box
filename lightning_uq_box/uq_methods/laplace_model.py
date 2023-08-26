"""Laplace Approximation model."""

import copy
import os
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
from laplace.curvature import AsdlGGN
from laplace.utils import LargestMagnitudeSubnetMask, ModuleNameSubnetMask
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.trainers.utils import _get_input_layer_name_and_module
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from tqdm import trange

from lightning_uq_box.eval_utils import compute_quantiles_from_std

from .utils import (
    _get_output_layer_name_and_module,
    change_inplace_activation,
    map_stochastic_modules,
    save_predictions_to_csv,
)


class LaplaceModel(LightningModule):
    """Laplace Approximation method for regression."""

    subset_of_weights_options = ["last_layer", "subnetwork", "all"]
    hessian_structure_options = ["diag", "kron", "full", "lowrank"]

    def __init__(
        self,
        model: nn.Module,
        save_dir: str,
        sigma_noise: float = 1.0,
        prior_precision: float = 1.0,
        prior_mean: float = 0.0,
        temperature: float = 1.0,
        subset_of_weights: str = "last_layer",
        hessian_structure: str = "kron",
        tune_precision_lr: float = 0.1,
        n_epochs_tune_precision: int = 100,
        n_params_subnet: int = None,
        part_stoch_module_names: Optional[list[Union[int, str]]] = None,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """Initialize a new instance of Laplace Model Wrapper.

        Args:
            model: pytorch model to use as underlying model
            laplace_args: laplace arguments to initialize a Laplace Model
            train_loader: train loader to be used but maybe this can
                also be accessed through the trainer or write a
                train_dataloader() method for this model based on the config?
        """
        super().__init__()

        if part_stoch_module_names and n_params_subnet:
            raise ValueError(
                "Only one of `part_stoch_module_names` or `n_params_subnet` can be defined."
            )

        if part_stoch_module_names or n_params_subnet:
            assert subset_of_weights == "subnetwork"

        self.save_hyperparameters(ignore=["model", "train_loader"])

        self.model = model  # pytorch model
        self.laplace_fitted = False

        self.pred_file_name = "predictions.csv"

        self.test_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(),
            },
            prefix="test_",
        )

        self.loss_fn = torch.nn.MSELoss()

    @property
    def num_inputs(self) -> int:
        """Retrieve input dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_input_layer_name_and_module(self.model)
        if hasattr(module, "in_features"):  # Linear Layer
            num_inputs = module.in_features
        elif hasattr(module, "in_channels"):  # Conv Layer
            num_inputs = module.in_channels
        return num_inputs

    @property
    def num_outputs(self) -> int:
        """Retrieve output dimension to the model.

        Returns:
            number of input dimension to the model
        """
        _, module = _get_output_layer_name_and_module(self.model)
        if hasattr(module, "out_features"):  # Linear Layer
            num_outputs = module.out_features
        elif hasattr(module, "out_channels"):  # Conv Layer
            num_outputs = module.out_channels
        return num_outputs

    def forward(self, X: Tensor, **kwargs: Any) -> np.ndarray:
        """Fitted Laplace Model Forward Pass.

        Args:
            X: tensor of data to run through the model [batch_size, input_dim]

        Returns:
            output from the laplace model
        """
        if not self.laplace_fitted:
            self.on_test_start()

        return self.la_model(X)

    def on_test_start(self) -> None:
        """Fit the Laplace approximation before testing."""
        self.train_loader = self.trainer.datamodule.train_dataloader()

        def collate_fn_laplace_torch(batch):
            """Collate function to for laplace torch tuple convention.

            Args:
                batch: input batch

            Returns:
                renamed batch
            """
            # Extract images and labels from the batch dictionary
            if isinstance(batch[0], dict):
                try:
                    images = [item["image"] for item in batch]
                    labels = [item["labels"] for item in batch]
                except KeyError:
                    images = [item["inputs"] for item in batch]
                    labels = [item["targets"] for item in batch]
            else:
                images = [item[0] for item in batch]
                labels = [item[1] for item in batch]

            # Stack images and labels into tensors
            inputs = torch.stack(images)
            targets = torch.stack(labels)

            # apply datamodule augmentation
            aug_batch = self.trainer.datamodule.on_after_batch_transfer(
                {"inputs": inputs, "targets": targets}, dataloader_idx=0
            )

            return (aug_batch["inputs"], aug_batch["targets"])

        self.train_loader.collate_fn = collate_fn_laplace_torch

        if not self.laplace_fitted:
            # select by largest params
            if self.hparams.n_params_subnet:
                assert self.hparams.subset_of_weights == "subnetwork"
                subnetwork_mask = LargestMagnitudeSubnetMask(
                    copy.deepcopy(self.model),
                    n_params_subnet=self.hparams.n_params_subnet,
                )
                subnetwork_indices = subnetwork_mask.select().cpu()
                del subnetwork_mask
                self.model.apply(change_inplace_activation)
            # select by module names
            elif self.hparams.part_stoch_module_names:
                assert self.hparams.subset_of_weights == "subnetwork"
                self.part_stoch_module_names = map_stochastic_modules(
                    self.model, part_stoch_module_names
                )
                subnetwork_mask = ModuleNameSubnetMask(
                    copy.deepcopy(self.model), module_names=self.part_stoch_module_names
                )
                subnetwork_mask.select()
                subnetwork_indices = subnetwork_mask.indices.cpu()
                del subnetwork_mask
                self.model.apply(change_inplace_activation)
            # last layer case
            else:
                subnetwork_indices = None

            # take the deterministic model we trained and fit laplace
            # laplace needs a nn.Module ant not a lightning module

            # also lightning automatically disables gradient computation during test
            # but need it for laplace so set inference mode to false with cntx manager
            with torch.inference_mode(False):
                self.la_model = Laplace(
                    model=self.model.model,
                    likelihood="regression",
                    subset_of_weights=self.hparams.subset_of_weights,
                    hessian_structure=self.hparams.hessian_structure,
                    # subnetwork_indices=subnetwork_indices,
                    sigma_noise=self.hparams.sigma_noise,
                    prior_precision=self.hparams.prior_precision,
                    prior_mean=self.hparams.prior_mean,
                    temperature=self.hparams.temperature,
                    backend=AsdlGGN,
                )
                # fit the laplace approximation
                self.la_model.fit(self.train_loader)

                # tune the prior precision via Empirical Bayes
                log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
                    1, requires_grad=True
                )
                hyper_optimizer = torch.optim.Adam(
                    [log_prior, log_sigma], lr=self.hparams.tune_precision_lr
                )
                bar = trange(self.hparams.n_epochs_tune_precision)
                # find out why this is so extremely slow?
                for i in bar:
                    hyper_optimizer.zero_grad()
                    neg_marglik = -self.la_model.log_marginal_likelihood(
                        log_prior.exp(), log_sigma.exp()
                    )
                    neg_marglik.backward()
                    hyper_optimizer.step()
                    bar.set_postfix(neg_marglik=f"{neg_marglik.detach().cpu().item()}")

            self.laplace_fitted = True

        # save this laplace fitted model as a checkpoint?!

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test step."""
        out_dict = self.predict_step(batch["inputs"])
        out_dict["targets"] = batch["targets"].detach().squeeze(-1).cpu().numpy()

        self.log(
            "test_loss", self.loss_fn(out_dict["pred"], batch["targets"].squeeze(-1))
        )  # logging to Logger
        if batch["inputs"].shape[0] > 1:
            self.test_metrics(out_dict["pred"], batch["targets"].squeeze(-1))

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1).numpy()

        # save metadata
        for key, val in batch.items():
            if key not in ["inputs", "targets"]:
                out_dict[key] = val.detach().squeeze(-1).cpu().numpy()

        return out_dict

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        if self.hparams.save_dir:
            save_predictions_to_csv(
                outputs, os.path.join(self.hparams.save_dir, self.pred_file_name)
            )

    def on_test_epoch_end(self):
        """Log epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Predict step with Laplace Approximation.

        Args:
            X: prediction batch of shape [batch_size x input_dims]

        Returns:
            prediction dictionary
        """
        if not self.laplace_fitted:
            self.on_test_start()

        # also lightning automatically disables gradient computation during test
        # but need it for laplace so set inference mode to false with context manager
        with torch.inference_mode(False):
            # inference tensors are not saved for backward so need to create
            # a clone with autograd enables
            input = X.clone().requires_grad_()

            laplace_mean, laplace_var = self.forward(input)
            laplace_mean = laplace_mean.squeeze().detach().cpu().numpy()
            laplace_epistemic = laplace_var.squeeze().sqrt().cpu().numpy()
            laplace_aleatoric = (
                np.ones_like(laplace_epistemic) * self.la_model.sigma_noise.item()
            )
            laplace_predictive = np.sqrt(
                laplace_epistemic**2 + laplace_aleatoric**2
            )
            quantiles = compute_quantiles_from_std(
                laplace_mean, laplace_predictive, self.hparams.quantiles
            )

        return {
            "pred": torch.from_numpy(laplace_mean).to(self.device),
            "pred_uct": laplace_predictive,
            "epistemic_uct": laplace_epistemic,
            "aleatoric_uct": laplace_aleatoric,
            "lower_quant": quantiles[:, 0],
            "upper_quant": quantiles[:, -1],
        }
