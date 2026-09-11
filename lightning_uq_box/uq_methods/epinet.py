# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Epistemic Neural Networks (epinet)."""

import os
from typing import Any, ClassVar

import torch
from einops import rearrange, repeat
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn

from lightning_uq_box.models.epinet import Epinet

from .base import DeterministicModel
from .loss_functions import EpinetGaussianNoiseLoss
from .utils import (
    _get_num_outputs,
    _get_output_layer_name_and_module,
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    process_regression_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


class EpinetBase(DeterministicModel):
    r"""Epistemic Neural Network (epinet) base class.

    An epinet supplements a conventional base network :math:`\mu_\zeta(x)` with a
    small head :math:`\sigma_\eta` that additionally consumes an epistemic index
    :math:`z \sim \mathcal{N}(0, I_{D_Z})`,

    .. math::
        f_\theta(x, z) = \mu_\zeta(x) + \sigma_\eta(\mathrm{sg}[\phi_\zeta(x)], z),

    where :math:`\mathrm{sg}` is a stop-gradient and :math:`\phi_\zeta(x)` are the
    features of the base network's last hidden layer. Because the features are only
    read, never written, the base network is left structurally untouched: an epinet can
    be attached to a large pretrained model whose weights are frozen, which is the
    paper's headline use case.

    The features are captured with a ``forward_pre_hook`` on the base network's output
    layer, rather than by replacing that layer, so that pretrained checkpoints continue
    to load into the user's model unchanged.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2107.08924

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        index_dim: int = 8,
        num_index_samples: int = 8,
        num_pred_samples: int = 100,
        epinet_hidden_dims: list[int] | None = None,
        prior_hidden_dims: list[int] | None = None,
        epi_prior_scale: float = 0.0,
        input_prior_scale: float = 0.3,
        use_input_features: bool = True,
        prior_seed: int = 0,
        freeze_backbone: bool = False,
        input_prior: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        epinet_concat_index: bool = True,
    ) -> None:
        """Initialize a new instance of the Epinet base class.

        Args:
            model: the base network the epinet is attached to
            loss_fn: loss function used for optimization
            index_dim: dimension :math:`D_Z` of the epistemic index
            num_index_samples: number of index samples drawn per training batch. The
                batch is repeated once per sample, so this multiplies the effective
                batch size.
            num_pred_samples: number of index samples drawn at prediction time
            epinet_hidden_dims: hidden sizes of the learnable epinet, defaults to
                ``[15, 15]``
            prior_hidden_dims: hidden sizes of the prior networks, defaults to
                ``[5, 5]``
            epi_prior_scale: scale of the epinet's own frozen prior over the features.
                The Neural Testbed agent uses ``0.0`` and puts all prior variation in
                the input prior; the image experiments do the opposite.
            input_prior_scale: scale of the additive frozen prior over the raw input
            use_input_features: whether to concatenate the flattened raw input to the
                base features handed to the epinet, as the Neural Testbed does. Not
                supported for image inputs.
            prior_seed: seed used to initialize the frozen prior networks
            freeze_backbone: whether to freeze the *entire* base network, its head
                included, leaving only the epinet trainable
            input_prior: prior module over the raw network input, taking
                ``(x, index)``. Required for convolutional base networks, where the
                flattened input size cannot be inferred; pass a
                :class:`~lightning_uq_box.models.ConvEnsemblePriorFunction`. When
                ``None`` an MLP ensemble prior is built for MLP-shaped bases.
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            epinet_concat_index: concatenate the index to the epinet MLP input;
                disable for a head that is linear in the index
        """
        if index_dim < 1 or num_index_samples < 1 or num_pred_samples < 2:
            raise ValueError(
                "index_dim and num_index_samples must be positive; "
                "num_pred_samples must be at least 2."
            )
        # plain attributes rather than self.hparams, which does not survive
        # the DeterministicModel init reliably
        self.index_dim = index_dim
        self.num_index_samples = num_index_samples
        self.num_pred_samples = num_pred_samples
        self.use_input_features = use_input_features

        # Infer widths before initialization; register the hook only after validation.
        self._features: Tensor | None = None
        n_feature_inputs, n_raw_inputs = self._feature_dimensions(
            model, use_input_features
        )

        # For a conv base the flattened input size is unknown, so an MLP prior over
        # the raw input cannot be sized. Say so here rather than failing later with a
        # matrix-shape error from inside the prior's forward pass.
        if n_raw_inputs == 0 and input_prior is None and input_prior_scale != 0.0:
            raise ValueError(
                "input_prior_scale is non-zero but no input_prior was given, and the "
                "flattened input size of a convolutional base network cannot be "
                "inferred. Pass a ConvEnsemblePriorFunction as input_prior, or set "
                "input_prior_scale=0.0 and rely on epi_prior_scale instead."
            )

        # built before super().__init__ so that its width is known, but assigned
        # after, since nn.Module forbids submodule assignment before its own init
        epinet = Epinet(
            n_feature_inputs=n_feature_inputs,
            n_raw_inputs=n_raw_inputs,
            n_outputs=_get_num_outputs(model),
            index_dim=index_dim,
            hidden_dims=epinet_hidden_dims,
            prior_hidden_dims=prior_hidden_dims,
            epi_prior_scale=epi_prior_scale,
            input_prior_scale=input_prior_scale,
            seed=prior_seed,
            input_prior=input_prior,
            concat_index=epinet_concat_index,
        )

        super().__init__(model, loss_fn, freeze_backbone, optimizer, lr_scheduler)

        self.epinet = epinet
        _, last_layer = _get_output_layer_name_and_module(model)
        self._feature_hook = last_layer.register_forward_pre_hook(
            self._capture_features
        )
        self.save_hyperparameters(
            ignore=["model", "loss_fn", "input_prior", "optimizer", "lr_scheduler"]
        )

    def _capture_features(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        """Capture detached features without retaining the base computation graph."""
        self._features = args[0].detach()

    def train(self, mode: bool = True) -> "EpinetBase":
        """Keep a frozen backbone's BatchNorm and dropout in evaluation mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def _feature_dimensions(
        self, model: nn.Module, use_input_features: bool
    ) -> tuple[int, int]:
        """Infer feature and raw input widths without mutating the base network.

        Args:
            model: the base network
            use_input_features: whether the raw input is concatenated to the features

        Returns:
            the epinet's feature input width and the flattened raw input width
        """
        _, last_layer = _get_output_layer_name_and_module(model)

        if hasattr(last_layer, "in_features"):
            n_feature_inputs = int(last_layer.in_features)
        elif hasattr(last_layer, "in_channels"):
            n_feature_inputs = int(last_layer.in_channels)
        else:
            raise ValueError(
                f"Output layer {last_layer} has neither in_features nor in_channels, "
                "so the epinet cannot determine its input width."
            )

        # Whether the *input* is an image is decided by the first parameterized layer,
        # not the last: a conv backbone with a linear classifier head (every ResNet)
        # still takes images, and its flattened input size is not in the module tree.
        takes_image_input = False
        for module in model.modules():
            if isinstance(module, nn.Conv2d | nn.Conv3d):
                takes_image_input = True
                break
            if isinstance(module, nn.Linear):
                break

        if use_input_features and takes_image_input:
            raise ValueError(
                "use_input_features=True flattens the raw network input into the "
                "epinet, which is not supported for image inputs. Set "
                "use_input_features=False, and use a ConvEnsemblePriorFunction as the "
                "input prior instead."
            )

        # The raw input width is only needed to size a default MLP prior over the
        # input. For an MLP-shaped base it is the first linear layer's input width;
        # for a conv base the flattened image size is not knowable from the module
        # tree, so it is left at 0 and the caller must supply its own prior.
        n_raw_inputs = 0
        if not takes_image_input:
            for module in model.modules():
                if hasattr(module, "in_features"):
                    n_raw_inputs = int(module.in_features)
                    break

        if use_input_features:
            n_feature_inputs += n_raw_inputs

        return n_feature_inputs, n_raw_inputs

    def freeze_model(self) -> None:
        """Freeze the whole base network, head included.

        Deliberately *not* the inherited ``freeze_model_backbone``, which freezes
        everything and then re-enables gradients on the head. The epinet's premise is
        that the base network is fixed and the epinet alone supplies the epistemic
        uncertainty, so the base head must stay frozen too.
        """
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def sample_index(self, num_samples: int, device: torch.device) -> Tensor:
        r"""Draw epistemic indices from the reference distribution.

        Args:
            num_samples: how many indices to draw
            device: device to place the indices on

        Returns:
            indices of shape [num_samples, index_dim], drawn from
            :math:`\mathcal{N}(0, I_{D_Z})`
        """
        return torch.randn(num_samples, self.index_dim, device=device, dtype=self.dtype)

    def extract_features(self, X: Tensor) -> Tensor:
        r"""Build the stop-gradiented feature vector handed to the epinet.

        Must be called after a forward pass through the base network, which is what
        populates the hook's capture.

        Args:
            X: the raw input the base network was run on

        Returns:
            features of shape [batch_size, n_feature_inputs]
        """
        if self._features is None:
            raise RuntimeError(
                "No features were captured. The base network must be run before the "
                "epinet features are read."
            )
        features = self._features.detach()
        if features.ndim > 2:
            features = features.flatten(2).mean(-1)
        if self.use_input_features:
            features = torch.cat([features, torch.flatten(X, 1).detach()], dim=-1)
        return features

    def forward(self, X: Tensor, index: Tensor | None = None) -> Tensor:
        r"""Forward pass of the combined network.

        Args:
            X: input tensor of shape [batch_size, \*input_shape]
            index: optional epistemic index of shape [batch_size, index_dim]. When
                ``None``, one index is drawn per example.

        Returns:
            output of shape [batch_size, num_outputs]
        """
        logits = self._base_forward(X)
        features = self.extract_features(X)
        if index is None:
            index = self.sample_index(X.shape[0], X.device)
        return logits + self.epinet(features, X.detach(), index)

    def _base_forward(self, X: Tensor) -> Tensor:
        """Run the base and reject outputs outside the vector prediction contract."""
        self._features = None
        logits = self.model(X)
        if logits.ndim != 2:
            raise ValueError(
                "Epinet requires base outputs of shape [batch_size, num_outputs]."
            )
        return logits

    def _loss(
        self, out: Tensor, target: Tensor, index: Tensor, data_index: Tensor | None
    ) -> Tensor:
        """Score repeated outputs, optionally using persistent bootstrap signatures."""
        return self.loss_fn(out, target)

    def compute_loss(
        self, X: Tensor, y: Tensor, data_index: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the base network once and the epinet over several indices.

        The base network is evaluated a single time and its output and features are
        repeated ``num_index_samples`` times, so the cost of averaging the loss over
        indices falls only on the small epinet head.

        Args:
            X: input tensor of shape [batch_size, ...]
            y: target tensor of shape [batch_size, ...]
            data_index: stable training example IDs of shape [batch_size], required
                only for Gaussian bootstrap regression

        Returns:
            the loss, the combined outputs and the repeated targets
        """
        batch_size = X.shape[0]
        num_samples = self.num_index_samples

        logits = self._base_forward(X)
        features = self.extract_features(X)

        # one index per sample, shared across the batch, following the paper
        index = self.sample_index(num_samples, X.device)

        # estimator-major "(k b)" ordering, matching MasksemblesBase
        features_k = repeat(features, "b f -> (k b) f", k=num_samples)
        x_k = repeat(X.detach(), "b ... -> (k b) ...", k=num_samples)
        index_k = repeat(index, "k d -> (k b) d", b=batch_size)
        logits_k = repeat(logits, "b c -> (k b) c", k=num_samples)

        out = logits_k + self.epinet(features_k, x_k, index_k)
        y_k = repeat(y, "b ... -> (k b) ...", k=num_samples)

        return self._loss(out, y_k, index_k, data_index), out, y_k

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        X, y = batch[self.input_key], batch[self.target_key]
        loss, out, y_k = self.compute_loss(X, y, batch.get("index"))

        # the index-sample repeat means the effective batch is larger than the input
        self.log("train_loss", loss, batch_size=X.shape[0] * self.num_index_samples)
        if X.shape[0] > 1:
            self.train_metrics(self.adapt_output_for_metrics(out), y_k)

        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        X, y = batch[self.input_key], batch[self.target_key]
        loss, out, y_k = self.compute_loss(X, y)

        self.log("val_loss", loss, batch_size=X.shape[0] * self.num_index_samples)
        if X.shape[0] > 1:
            self.val_metrics(self.adapt_output_for_metrics(out), y_k)

        return loss

    def predict_samples(self, X: Tensor, indices: Tensor | None = None) -> Tensor:
        r"""Draw ENN samples for a batch of inputs.

        Reuse the same ``indices`` for every batch of an evaluation set before
        concatenating samples for joint metrics. Independently resampling indices
        for each batch destroys the dependence between predictions.

        Args:
            X: input tensor of shape [batch_size, \*input_shape]
            indices: optional shared indices of shape [num_samples, index_dim]

        Returns:
            samples of shape [batch_size, num_outputs, num_samples], the layout
            the ``process_*_prediction`` helpers expect
        """
        batch_size = X.shape[0]
        if indices is None:
            indices = self.sample_index(self.num_pred_samples, X.device)
        if (
            indices.ndim != 2
            or indices.shape[1] != self.index_dim
            or indices.shape[0] < 1
        ):
            raise ValueError("indices must have shape [num_samples, index_dim].")
        num_samples = indices.shape[0]

        with torch.no_grad():
            logits = self._base_forward(X)
            features = self.extract_features(X)

            index = indices.to(device=X.device, dtype=features.dtype)
            features_k = repeat(features, "b f -> (k b) f", k=num_samples)
            x_k = repeat(X, "b ... -> (k b) ...", k=num_samples)
            index_k = repeat(index, "k d -> (k b) d", b=batch_size)
            logits_k = repeat(logits, "b c -> (k b) c", k=num_samples)

            out = logits_k + self.epinet(features_k, x_k, index_k)

        return rearrange(out, "(k b) c -> b c k", k=num_samples)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Initialize the optimizer and learning rate scheduler.

        Only parameters that require gradients are handed to the optimizer, so the
        frozen prior networks never receive an update or a weight-decay penalty, and a
        frozen base network is excluded as well.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optimizer(params)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        return {"optimizer": optimizer}


class EpinetRegression(EpinetBase):
    """Epinet for regression tasks.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2107.08924

    .. versionadded:: 0.4
    """

    pred_file_name = "preds.csv"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        index_dim: int = 8,
        num_index_samples: int = 8,
        num_pred_samples: int = 100,
        epinet_hidden_dims: list[int] | None = None,
        prior_hidden_dims: list[int] | None = None,
        epi_prior_scale: float = 0.0,
        input_prior_scale: float = 0.3,
        use_input_features: bool = True,
        prior_seed: int = 0,
        freeze_backbone: bool = False,
        input_prior: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        epinet_concat_index: bool = True,
        bootstrap_noise_scale: float = 0.0,
        num_train: int | None = None,
    ) -> None:
        """Initialize regression, optionally with equation 9 Gaussian bootstrapping.

        Args:
            model: base regression network
            loss_fn: unperturbed regression loss; bootstrap training uses squared error
            index_dim: dimension of the Gaussian index
            num_index_samples: shared index draws per training batch
            num_pred_samples: prediction draws
            epinet_hidden_dims: epinet hidden widths
            prior_hidden_dims: input prior hidden widths
            epi_prior_scale: feature prior multiplier
            input_prior_scale: raw input prior multiplier
            use_input_features: include raw inputs in the epinet features
            prior_seed: fixed prior and bootstrap signature seed
            freeze_backbone: freeze the entire base including its running statistics
            input_prior: optional frozen prior over raw inputs
            optimizer: optimizer factory
            lr_scheduler: scheduler factory
            epinet_concat_index: include the index in the MLP input
            bootstrap_noise_scale: observation standard deviation; zero disables bootstrap
            num_train: number of stable training IDs, required when bootstrap is enabled
        """
        if bootstrap_noise_scale < 0:
            raise ValueError("bootstrap_noise_scale must be nonnegative.")
        if bootstrap_noise_scale > 0 and (num_train is None or num_train < 1):
            raise ValueError(
                "Gaussian bootstrap requires num_train > 0 and stable batch 'index' IDs."
            )
        self.bootstrap_noise_scale = bootstrap_noise_scale
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            index_dim=index_dim,
            num_index_samples=num_index_samples,
            num_pred_samples=num_pred_samples,
            epinet_hidden_dims=epinet_hidden_dims,
            prior_hidden_dims=prior_hidden_dims,
            epi_prior_scale=epi_prior_scale,
            input_prior_scale=input_prior_scale,
            use_input_features=use_input_features,
            prior_seed=prior_seed,
            freeze_backbone=freeze_backbone,
            input_prior=input_prior,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epinet_concat_index=epinet_concat_index,
        )
        if bootstrap_noise_scale > 0 and _get_num_outputs(model) != 1:
            raise ValueError("Gaussian bootstrap supports a scalar regression output.")
        generator = torch.Generator().manual_seed(prior_seed)
        signatures = torch.randn(num_train or 0, index_dim, generator=generator)
        self.register_buffer(
            "bootstrap_signatures", nn.functional.normalize(signatures, dim=-1)
        )
        self.bootstrap_loss = EpinetGaussianNoiseLoss(bootstrap_noise_scale)

    bootstrap_signatures: Tensor

    def _loss(
        self, out: Tensor, target: Tensor, index: Tensor, data_index: Tensor | None
    ) -> Tensor:
        """Use fixed per-example signatures only during bootstrap training."""
        if self.bootstrap_noise_scale == 0 or not self.training:
            return self.loss_fn(out, target)
        if data_index is None:
            raise ValueError(
                "Gaussian bootstrap requires stable example IDs in batch['index']."
            )
        batch_size = out.shape[0] // self.num_index_samples
        if data_index.shape != (batch_size,) or data_index.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("batch['index'] must contain one integer ID per example.")
        if (data_index < 0).any() or (
            data_index >= self.bootstrap_signatures.shape[0]
        ).any():
            raise ValueError("batch['index'] IDs must lie in [0, num_train).")
        signature = self.bootstrap_signatures[data_index]
        signature = repeat(signature, "b d -> (k b) d", k=self.num_index_samples)
        return self.bootstrap_loss(out, target, signature, index)

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            mean output
        """
        return out[:, 0:1]

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Prediction step.

        Args:
            X: prediction batch of shape [batch_size, \*input_shape]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            dictionary with the mean prediction, the predictive uncertainty and the
            raw ENN samples
        """
        samples = self.predict_samples(X)
        pred_dict = process_regression_prediction(samples)
        pred_dict["samples"] = samples
        return pred_dict

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class EpinetClassification(EpinetBase):
    """Epinet for classification tasks.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2107.08924

    .. versionadded:: 0.4
    """

    pred_file_name = "preds.csv"

    valid_tasks: ClassVar[list[str]] = ["binary", "multiclass", "multilabel"]

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        task: str = "multiclass",
        index_dim: int = 8,
        num_index_samples: int = 8,
        num_pred_samples: int = 100,
        epinet_hidden_dims: list[int] | None = None,
        prior_hidden_dims: list[int] | None = None,
        epi_prior_scale: float = 0.0,
        input_prior_scale: float = 0.3,
        use_input_features: bool = True,
        prior_seed: int = 0,
        freeze_backbone: bool = False,
        input_prior: nn.Module | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable | None = None,
        epinet_concat_index: bool = True,
    ) -> None:
        """Initialize a new instance of the Epinet classification model.

        Args:
            model: the base network the epinet is attached to
            loss_fn: loss function used for optimization
            task: what kind of classification task, choose one of
                ["binary", "multiclass", "multilabel"]
            index_dim: dimension of the epistemic index
            num_index_samples: number of index samples drawn per training batch
            num_pred_samples: number of index samples drawn at prediction time
            epinet_hidden_dims: hidden sizes of the learnable epinet
            prior_hidden_dims: hidden sizes of the prior networks
            epi_prior_scale: scale of the epinet's own frozen prior over the features
            input_prior_scale: scale of the additive frozen prior over the raw input
            use_input_features: whether to concatenate the flattened raw input to the
                base features handed to the epinet
            prior_seed: seed used to initialize the frozen prior networks
            freeze_backbone: whether to freeze the entire base network
            input_prior: prior module over the raw network input, required for
                convolutional base networks
            optimizer: optimizer used for training
            lr_scheduler: learning rate scheduler
            epinet_concat_index: concatenate the index to the epinet MLP input;
                disable for a head that is linear in the index
        """
        # set before super().__init__, which calls setup_task
        self.num_classes = _get_num_outputs(model)
        assert task in self.valid_tasks, f"Task must be one of {self.valid_tasks}"
        self.task = task
        super().__init__(
            model,
            loss_fn,
            index_dim,
            num_index_samples,
            num_pred_samples,
            epinet_hidden_dims,
            prior_hidden_dims,
            epi_prior_scale,
            input_prior_scale,
            use_input_features,
            prior_seed,
            freeze_backbone,
            input_prior,
            optimizer,
            lr_scheduler,
            epinet_concat_index=epinet_concat_index,
        )

    def setup_task(self) -> None:
        """Set up task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def adapt_output_for_metrics(self, out: Tensor) -> Tensor:
        """Adapt model output to be compatible for metric computation.

        Args:
            out: output from the model

        Returns:
            the output unchanged
        """
        return out

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        r"""Prediction step.

        Args:
            X: prediction batch of shape [batch_size, \*input_shape]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            dictionary with the mean class probabilities, the predictive uncertainty
            and the raw ENN logits
        """
        samples = self.predict_samples(X)
        return process_classification_prediction(samples, task=self.task)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch: batch from dataloader
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        save_classification_predictions(
            outputs,
            os.path.join(self.trainer.default_root_dir, self.pred_file_name),
            task=self.task,
        )
