# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

# Adapted for Lightning from https://github.com/VectorInstitute/vbll

"""Variational Bayesian Last Layer (VBLL)."""

from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torch.nn.modules import Module

from .base import DeterministicClassification, DeterministicRegression
from .utils import (
    _get_output_layer_name_and_module,
    default_classification_metrics,
    replace_module,
)


class VBLLRegression(DeterministicRegression):
    """Variational Bayesian Last Layer (VBLL) for Regression.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2404.11599

    """

    def __init__(
        self,
        model: Module,
        regularization_weight,
        replace_ll: bool = True,
        num_targets: int = 1,
        parameterization: str = "dense",
        prior_scale: float = 1.0,
        wishart_scale: float = 1e-2,
        dof: int = 1,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
    ) -> None:
        """Initialize the VBLL regression model.

        Args:
            model: The backbone model
            regularization_weight : regularization weight term in ELBO, and should be
                1 / (dataset size) by default. This term impacts the epistemic
                uncertainty estimate.
            replace_ll: Whether to replace the last layer of the model with VBLL
                or add a new layer
            num_targets : Number of targets
            parameterization : Parameterization of covariance matrix. One of
                ['dense','diagonal']
            prior_scale : prior covariance matrix scale
                Scale of prior covariance matrix
            wishart_scale : Scale of Wishart prior on noise covariance. This term
                has an impact on the aleatoric uncertainty estimate.
            dof : Degrees of freedom of Wishart prior on noise covariance
            freeze_backbone: If True, the backbone model will be frozen
                and only the VBBL layer will be trained
            optimizer: The optimizer to use for training
            lr_scheduler: The learning rate scheduler to use for training

        """
        # pass freeze model False as we will freeze the backbone in the model below customly
        super().__init__(model, None, freeze_backbone, optimizer, lr_scheduler)

        try:
            import vbll  # noqa: F401
        except ImportError:
            raise ImportError(
                "You need to install the vbll package: 'pip install vbll'."
            )

        self.regularization_weight = regularization_weight
        self.num_targets = num_targets
        self.parameterization = parameterization
        self.prior_scale = prior_scale
        self.wishart_scale = wishart_scale
        self.dof = dof
        self.replace_ll = replace_ll
        self.build_model()
        self.freeze_model()

    def build_model(self) -> None:
        """Build model."""
        from vbll import Regression as VBLLReg

        last_layer_name, last_module_backbone = _get_output_layer_name_and_module(
            self.model
        )

        if self.replace_ll:
            in_features = last_module_backbone.in_features
        else:
            in_features = last_module_backbone.out_features

        new_layer = VBLLReg(
            in_features=in_features,
            out_features=self.num_targets,
            regularization_weight=self.regularization_weight,
            parameterization=self.parameterization,
            prior_scale=self.prior_scale,
            wishart_scale=self.wishart_scale,
            dof=self.dof,
        )

        if self.replace_ll:
            replace_module(self.model, last_layer_name, new_layer)
        else:
            self.model = nn.Sequential(self.model, new_layer)

    def freeze_model(self) -> None:
        """Freeze model."""
        if self.freeze_backbone:
            for name, module in self.model.named_modules():
                if module.__class__.__name__ == "Regression":
                    for param in module.parameters():
                        param.requires_grad = True
                elif not any(module.named_children()):
                    for param in module.parameters():
                        param.requires_grad = False

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for metrics.

        Args:
            out: the output from the VBLL module

        Returns:
            the mean prediction
        """
        return out.predictive.mean

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            training loss
        """
        out = self.model(batch[self.input_key])
        loss = out.train_loss_fn(batch[self.target_key])

        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            validation loss
        """
        out = self.model(batch[self.input_key])
        loss = out.val_loss_fn(batch[self.target_key])

        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            test loss
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key]

        test_loss = pred_dict["out"].val_loss_fn(batch[self.target_key])

        self.log("test_loss", test_loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(pred_dict["out"]), batch[self.target_key]
            )

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)
        # delete out from pred_dict
        del pred_dict["out"]
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step with VBLL model."""
        with torch.no_grad():
            pred = self.model(X)

        # TODO can we separate epistemic and aleatoric uncertainty of the
        # prediction?

        return {
            "pred": pred.predictive.mean,
            "pred_uct": torch.sqrt(pred.predictive.covariance).squeeze(-1),
            "out": pred,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Optimizers."""
        # exclude vbll parameters from weight decay
        backbone_params = []
        vbll_params = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "Regression":
                vbll_params.extend(list(module.parameters()))
            elif not any(module.named_children()):
                backbone_params.extend(list(module.parameters()))

        optimizer = self.optimizer(
            [{"params": backbone_params}, {"params": vbll_params, "weight_decay": 0.0}]
        )
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}


class VBLLClassification(DeterministicClassification):
    """Variational Bayes Last Layer (VBLL) for Classification.

    If you use this method in your research, please cite the following paper:

    * https://arxiv.org/abs/2404.11599
    """

    valid_layer_types = ["disc", "gen"]

    def __init__(
        self,
        model: nn.Module,
        regularization_weight: float,
        num_targets: int,
        replace_ll: bool = True,
        parameterization: str = "dense",
        prior_scale: float = 1,
        wishart_scale: float = 0.01,
        dof: int = 1,
        layer_type: str = "disc",
        freeze_backbone: bool = False,
        task: "str" = "multiclass",
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Any | None = None,
    ) -> None:
        """Initialize a new instance of VBLL Classification.

        Args:
            model: The backbone model
            regularization_weight : regularization weight term in ELBO, and should be
                1 / (dataset size) by default. This term impacts the epistemic
                uncertainty estimate.
            num_targets : Number of targets
            replace_ll: If True, replace the last layer of the model with VBLL
                or add a new layer
            parameterization : Parameterization of covariance matrix. One of
                ['dense','diagonal']
            prior_scale : prior covariance matrix scale
                Scale of prior covariance matrix
            wishart_scale : Scale of Wishart prior on noise covariance. This term
                has an impact on the aleatoric uncertainty estimate.
            dof : Degrees of freedom of Wishart prior on noise covariance
            layer_type: The type of layer to use. One of ['disc', 'gen'], a
                Discriminative or Generative layer
            freeze_backbone: If True, the backbone model will be frozen
                and only the VBBL layer will be trained
            task: The type of task. One of ['binary', 'multiclass']
            optimizer: The optimizer to use for training
            lr_scheduler: The learning rate scheduler to use for training
        """
        try:
            import vbll  # noqa: F401
        except ImportError:
            raise ImportError(
                "You need to install the vbll package: 'pip install vbll'."
            )

        self.num_targets = num_targets

        assert (
            layer_type in self.valid_layer_types
        ), f"layer_type must be one of {self.valid_layer_types}"

        if layer_type == "gen":
            assert (
                parameterization == "diagonal"
            ), "parameterization must be 'diagonal' for Generative layer"
        self.layer_type = layer_type

        # pass freeze model False as we will freeze the backbone in the model below customly
        super().__init__(model, None, task, False, optimizer, lr_scheduler)

        self.freeze_backbone = freeze_backbone

        self.regularization_weight = regularization_weight
        self.parameterization = parameterization
        self.prior_scale = prior_scale
        self.wishart_scale = wishart_scale
        self.dof = dof
        self.replace_ll = replace_ll

        self.build_model()
        self.freeze_model()

    def build_model(self) -> None:
        """Build Classification Model."""
        from vbll import DiscClassification as VBLLDiscClass
        from vbll import GenClassification as VBLLGenClass

        last_layer_name, last_module_backbone = _get_output_layer_name_and_module(
            self.model
        )

        new_layer = VBLLDiscClass if self.layer_type == "disc" else VBLLGenClass

        if self.replace_ll:
            in_features = last_module_backbone.in_features
        else:
            in_features = last_module_backbone.out_features

        new_layer = new_layer(
            in_features=in_features,
            out_features=self.num_targets,
            regularization_weight=self.regularization_weight,
            parameterization=self.parameterization,
            prior_scale=self.prior_scale,
            wishart_scale=self.wishart_scale,
            dof=self.dof,
            return_ood=True,
        )

        if self.replace_ll:
            replace_module(self.model, last_layer_name, new_layer)
        else:
            self.model = nn.Sequential(self.model, new_layer)

        self.num_classes = self.num_targets

    def freeze_model(self) -> None:
        """Freeze model."""
        if self.freeze_backbone:
            for name, module in self.model.named_modules():
                if module.__class__.__name__ in [
                    "DiscClassification",
                    "GenClassification",
                ]:
                    for param in module.parameters():
                        param.requires_grad = True
                elif not any(module.named_children()):
                    for param in module.parameters():
                        param.requires_grad = False

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_targets
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_targets
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_targets
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt the output for metrics."""
        return out.predictive.probs

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            training loss
        """
        out = self.model(batch[self.input_key])
        loss = out.train_loss_fn(batch[self.target_key])

        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.train_metrics(
                self.adapt_output_for_metrics(out), batch[self.target_key]
            )

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            validation loss
        """
        out = self.model(batch[self.input_key])
        loss = out.val_loss_fn(batch[self.target_key])

        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.val_metrics(self.adapt_output_for_metrics(out), batch[self.target_key])

        return loss

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            test loss
        """
        pred_dict = self.predict_step(batch[self.input_key])
        pred_dict[self.target_key] = batch[self.target_key]

        test_loss = pred_dict["out"].val_loss_fn(batch[self.target_key])

        self.log("test_loss", test_loss, batch_size=batch[self.input_key].shape[0])
        if batch[self.input_key].shape[0] > 1:
            self.test_metrics(
                self.adapt_output_for_metrics(pred_dict["out"]), batch[self.target_key]
            )

        pred_dict = self.add_aux_data_to_dict(pred_dict, batch)
        # delete out from pred_dict
        del pred_dict["out"]
        return pred_dict

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Predict step with VBLL model.

        Args:
            X: The input data
            batch_idx: The index of the batch
            dataloader_idx: The index of the dataloader

        Returns:
            prediction dictionary
        """
        with torch.no_grad():
            pred = self.model(X)

        probs = pred.predictive.probs

        entropy = -(probs * probs.log()).sum(dim=-1)

        return {
            "pred": probs,
            "pred_uct": entropy,
            "out": pred,
            "ood_score": pred.ood_scores,
            "class_probs": probs,
        }

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure Optimizers."""
        # exclude vbll parameters from weight decay
        backbone_params = []
        vbll_params = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ in ["DiscClassification", "GenClassification"]:
                vbll_params.extend(list(module.parameters()))
            elif not any(module.named_children()):
                backbone_params.extend(list(module.parameters()))

        optimizer = self.optimizer(
            [{"params": backbone_params}, {"params": vbll_params, "weight_decay": 0.0}]
        )
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
