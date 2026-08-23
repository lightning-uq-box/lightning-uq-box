# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utilities for UQ-Method Implementations."""

import json
import os
from collections import OrderedDict
from collections.abc import Callable

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import (
    Accuracy,
    CalibrationError,
    F1Score,
    JaccardIndex,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    R2Score,
)

from lightning_uq_box.eval_utils import (
    compute_aleatoric_uncertainty,
    compute_epistemic_uncertainty,
    compute_predictive_uncertainty,
    compute_quantiles_from_std,
)

from .metrics import EmpiricalCoverage


def checkpoint_loader(
    model_class: LightningModule, ckpt_path: str, return_model: bool = False
) -> LightningModule | nn.Module:
    """Load state dict checkpoint for LightningModule.

    Args:
        model_class: LightningModule class
        ckpt_path: path to checkpoint
        return_model: whether to return the model or the model class

    Returns:
        model_class or model
    """
    model_class.load_state_dict(
        state_dict=torch.load(ckpt_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    if return_model:
        return model_class.model
    else:
        return model_class


def identity(x: Tensor, dim: int | None = None) -> Tensor:
    """Return the input unchanged.

    Used as the aggregation function when there is no sample dimension to
    aggregate over.

    Args:
        x: input tensor
        dim: unused, accepted to match the aggregation function signature

    Returns:
        the input tensor
    """
    return x


def default_regression_metrics(prefix: str, include_r2: bool = True):
    """Return a set of default regression metrics.

    Args:
        prefix: prefix prepended to metric names.
        include_r2: include R² when a lifecycle guarantees at least two
            accumulated observations.  Canonical task runtimes leave it out so
            a valid one-item batch can be counted and completed.
    """
    metrics = {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()}
    if include_r2:
        metrics["R2"] = R2Score()
    return MetricCollection(metrics, prefix=prefix)


def default_px_regression_metrics(prefix: str):
    """Return a set of default regression metrics."""
    return MetricCollection(
        {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
        prefix=prefix,
    )


def default_classification_metrics(prefix: str, task: str, num_classes: int):
    """Return a set of default classification metrics.

    Args:
        prefix: prefix for the metric names
        task: one of "binary", "multiclass" or "multilabel"
        num_classes: number of classes, or number of labels for the
            "multilabel" task

    Returns:
        metric collection for the task
    """
    if task == "multilabel":
        # CalibrationError has no multilabel variant, and EmpiricalCoverage is
        # defined in terms of a single true label per sample, so neither is
        # part of the multilabel metric set.
        return MetricCollection(
            {
                "Acc": Accuracy(task=task, num_labels=num_classes),
                "F1Score": F1Score(task=task, num_labels=num_classes),
            },
            prefix=prefix,
        )
    if task == "binary":
        # EmpiricalCoverage is defined over a class-set axis and cannot be
        # updated from the one-logit ``[batch]`` representation.  Omitting it
        # here lets canonical one-logit BCE models count every batch,
        # including batch size one, rather than skipping those batches.
        return MetricCollection(
            {
                "Acc": Accuracy(task="binary"),
                "Calibration": CalibrationError(task="binary"),
            },
            prefix=prefix,
        )
    return MetricCollection(
        {
            "Acc": Accuracy(task=task, num_classes=num_classes),
            "Calibration": CalibrationError(task, num_classes=num_classes),
            "Empirical Coverage": EmpiricalCoverage(),
        },
        prefix=prefix,
    )


def default_segmentation_metrics(prefix: str, task: str, num_classes: int):
    """Return a set of default segmentation metrics.

    Args:
        prefix: prefix for the metric names
        task: one of "binary", "multiclass" or "multilabel"
        num_classes: number of classes, or number of labels for the
            "multilabel" task

    Returns:
        metric collection for the task
    """
    if task == "multilabel":
        return MetricCollection(
            {
                "Jaccard": JaccardIndex(task=task, num_labels=num_classes),
                "F1Score": F1Score(task=task, num_labels=num_classes),
            },
            prefix=prefix,
        )
    return MetricCollection(
        {
            "Jaccard": JaccardIndex(task=task, num_classes=num_classes),
            "F1Score": F1Score(task, num_classes=num_classes),
        },
        prefix=prefix,
    )


def process_regression_prediction(
    preds: Tensor,
    quantiles: list[float] | None = None,
    aggregate_fn: Callable = torch.mean,
) -> dict[str, Tensor]:
    """Process regression predictions that could be mse or nll predictions.

    Args:
        preds: prediction tensor of shape [batch_size, num_outputs, num_samples]
        quantiles: quantiles to compute
        aggregate_fn: function to aggregate over the samples to form a mean

    Returns:
        dictionary with mean prediction and predictive uncertainty
    """
    mean_samples = preds[:, 0, ...]
    mean = aggregate_fn(preds[:, 0:1, ...], dim=-1)
    # assume nll prediction with sigma
    if preds.shape[1] == 2:
        log_sigma_2_samples = preds[:, 1, ...]
        eps = torch.ones_like(log_sigma_2_samples) * 1e-6
        sigma_samples = torch.sqrt(eps + torch.exp(log_sigma_2_samples))
        std = compute_predictive_uncertainty(mean_samples, sigma_samples)
        aleatoric = compute_aleatoric_uncertainty(sigma_samples)
        epistemic = compute_epistemic_uncertainty(mean_samples)

        pred_dict = {
            "pred": mean,
            "pred_uct": std,
            "epistemic_uct": epistemic,
            "aleatoric_uct": aleatoric,
        }
    # assume mse prediction
    else:
        std = mean_samples.std(-1)
        pred_dict = {"pred": mean, "pred_uct": std, "epistemic_uct": std}

    # check if quantiles are present
    if quantiles is not None:
        quantiles = compute_quantiles_from_std(
            mean.detach().cpu().numpy(), std, quantiles
        )
        pred_dict["lower_quant"] = torch.from_numpy(quantiles[:, 0])
        pred_dict["upper_quant"] = torch.from_numpy(quantiles[:, -1])

    return pred_dict


def process_classification_prediction(
    logits: Tensor,
    aggregate_fn: Callable = torch.mean,
    eps: float = 1e-7,
    task: str = "multiclass",
    binary_encoding: str = "two_logit",
) -> dict[str, Tensor]:
    """Process classification predictions.

    Applies softmax to logit and computes mean over the samples and entropy.

    For the "multilabel" task the labels are not mutually exclusive, so a
    sigmoid is applied per label and the uncertainty is the sum of the
    per-label binary entropies.

    .. versionchanged:: 0.4.0

       Canonical callers can pass an explicit one-logit or two-logit binary
       encoding. The default remains two-logit to preserve legacy callers.

    Args:
        logits: prediction logits tensor of shape [batch_size, num_classes, num_samples]
        aggregate_fn: function to aggregate over the samples
        eps: small value to prevent log of 0
        task: one of "binary", "multiclass" or "multilabel"
        binary_encoding: binary probability encoding; canonical callers pass
            ``"one_logit"`` for BCE/sigmoid and ``"two_logit"`` for
            CE/softmax.  The legacy default preserves historical two-logit
            behavior.

    Returns:
        dictionary with aggregated class probabilities [batch_size, num_classes]
            and predictive uncertainty [batch_size]
    """
    if task == "multilabel":
        mean = aggregate_fn(torch.sigmoid(logits), dim=-1)
        mean = mean.clamp(eps, 1 - eps)
        entropy = -(mean * mean.log() + (1 - mean) * (1 - mean).log()).sum(dim=-1)
        return {"pred": mean, "pred_uct": entropy, "logits": logits}

    if task == "binary" and binary_encoding == "one_logit":
        mean = aggregate_fn(torch.sigmoid(logits), dim=-1)
        mean = mean.clamp(eps, 1 - eps)
        entropy = -(mean * mean.log() + (1 - mean) * (1 - mean).log())
        return {"pred": mean, "pred_uct": entropy.squeeze(1), "logits": logits}

    mean = aggregate_fn(nn.functional.softmax(logits, dim=1), dim=-1)
    # prevent log of 0 -> nan
    mean.clamp_min_(eps)
    entropy = -(mean * mean.log()).sum(dim=-1)
    return {"pred": mean, "pred_uct": entropy, "logits": logits}


def process_segmentation_prediction(
    logits: Tensor,
    aggregate_fn: Callable = torch.mean,
    eps: float = 1e-7,
    task: str = "multiclass",
    binary_encoding: str = "two_logit",
) -> dict[str, Tensor]:
    """Process segmentation predictions.

    Applies softmax to logit and computes mean over the samples and entropy.

    For the "multilabel" task the labels are not mutually exclusive, so a
    sigmoid is applied per label and the uncertainty is the sum of the
    per-label binary entropies.

    .. versionchanged:: 0.4.0

       Canonical callers can pass an explicit one-logit or two-logit binary
       encoding. The default remains two-logit to preserve legacy callers.

    Args:
        logits: prediction logits tensor of shape
            [batch_size, num_classes, height, width, num_samples]
        aggregate_fn: function to aggregate over the samples
        eps: small value to prevent log of 0
        task: one of "binary", "multiclass" or "multilabel"
        binary_encoding: binary probability encoding; canonical callers pass
            ``"one_logit"`` for BCE/sigmoid and ``"two_logit"`` for
            CE/softmax.  The legacy default preserves historical two-logit
            behavior.

    Returns:
        dictionary with pixel class probabilities
            [batch_size, num_classes, height, width]
        and predictive uncertainty [batch_size, height, width]
    """
    if task == "multilabel":
        mean = aggregate_fn(torch.sigmoid(logits), dim=-1)
        mean = mean.clamp(eps, 1 - eps)
        entropy = -(mean * mean.log() + (1 - mean) * (1 - mean).log()).sum(dim=1)
        return {"pred": mean, "pred_uct": entropy, "logits": logits}

    if task == "binary" and binary_encoding == "one_logit":
        mean = aggregate_fn(torch.sigmoid(logits), dim=-1)
        mean = mean.clamp(eps, 1 - eps)
        entropy = -(mean * mean.log() + (1 - mean) * (1 - mean).log()).squeeze(1)
        return {"pred": mean, "pred_uct": entropy, "logits": logits}

    mean = aggregate_fn(nn.functional.softmax(logits, dim=1), dim=-1)
    # prevent log of 0 -> nan
    mean.clamp_min_(eps)
    entropy = -(mean * mean.log()).sum(dim=1)
    return {"pred": mean, "pred_uct": entropy, "logits": logits}


def change_inplace_activation(module):
    """Change inplace activation."""
    if hasattr(module, "inplace"):
        module.inplace = False


def _distributed_path(path: str) -> str:
    """Return a rank-sharded path and publish a lightweight root manifest.

    Single-process callers keep their historical filename.  In distributed
    prediction each rank writes only to its own file/directory, avoiding the
    append and HDF5 collisions caused by a shared path.  The manifest is an
    intentionally simple discovery record; dataset-index information remains
    in the persisted ``index`` auxiliary field when a datamodule provides it.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return path
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return path
    rank = torch.distributed.get_rank()
    root, extension = os.path.splitext(path)
    sharded = f"{root}.rank-{rank}{extension}"
    if rank == 0:
        manifest_path = f"{root}.manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            json.dump(
                {
                    "version": 1,
                    "world_size": world_size,
                    "shards": [
                        os.path.basename(f"{root}.rank-{worker}{extension}")
                        for worker in range(world_size)
                    ],
                },
                manifest,
            )
    return sharded


def _writer_array(value: Tensor | object) -> np.ndarray:
    """Convert one writer value without deleting a one-item batch axis."""
    array = (
        value.detach().cpu().numpy() if isinstance(value, Tensor) else np.array(value)
    )
    if array.ndim >= 2 and array.shape[-1] == 1:
        return np.squeeze(array, axis=-1)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def save_image_predictions(outputs: STEP_OUTPUT, batch_idx: int, save_dir: str) -> None:
    """Save segmentation predictions to separate hdf5 files.

    .. versionchanged:: 0.4.0

       Distributed runs write rank-sharded dense predictions and a root
       manifest instead of allowing multiple ranks to overwrite one file.

    Args:
        outputs: metrics and values to be saved
            - pred: predictions of shape [batch_size, ...]
            - pred_uct: predictive uncertainty of shape [batch_size, ...]
            - target: targets of shape [batch_size, ...]
            - logits: logits of shape [batch_size, ...]
        batch_idx: index of the current batch
        save_dir: directory where hdf5 files should be saved
    """
    # Lightning types the hook argument as STEP_OUTPUT; the UQ methods always
    # return a dict from test_step, so narrow once here rather than at every hook.
    assert isinstance(outputs, dict)
    save_dir = _distributed_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for sample_idx in range(outputs["pred"].shape[0]):
        with h5py.File(
            f"{save_dir}/batch_{batch_idx}_sample_{sample_idx}.hdf5", "w"
        ) as f:
            for key, val in outputs.items():
                if isinstance(val, Tensor):
                    data = val[sample_idx].cpu().numpy()
                else:
                    data = np.array(val[sample_idx])
                if data.size == 1:  # single element array, save as attribute
                    f.attrs[key] = data.item()
                else:  # multi-element array, save as dataset
                    f.create_dataset(key, data=data, compression="gzip")


def save_regression_predictions(outputs: STEP_OUTPUT, path: str) -> None:
    """Save regression predictions to csv file.

    .. versionchanged:: 0.4.0

       The writer copies the output mapping and preserves one-item batch axes;
       it no longer consumes ``samples`` from the caller's result dictionary.

    Args:
        outputs: metrics and values to be saved
            - pred: predictions of shape [batch_size]
            - pred_uct: predictive uncertainty of shape [batch_size]
            - epistemic_uct: epistemic uncertainty of shape [batch_size]
            - aleatoric_uct: aleatoric uncertainty of shape [batch_size]
            - lower_quant: lower quantile of shape [batch_size]
            - upper_quant: upper quantile of shape [batch_size]
        path: path where csv should be saved
    """
    # Lightning types the hook argument as STEP_OUTPUT; the UQ methods always
    # return a dict from test_step, so narrow once here rather than at every hook.
    assert isinstance(outputs, dict)
    # Writers are observers: callers often use the same output object for
    # logging/callbacks after this hook, so never pop or otherwise mutate it.
    copied_outputs = dict(outputs)
    path = _distributed_path(path)
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    cpu_outputs = {}
    if "samples" in copied_outputs:
        samples = copied_outputs.pop("samples")
        for i in range(samples.shape[-1]):
            sample = _writer_array(samples[..., i])
            # mve prediction
            if sample.ndim == 2 and sample.shape[-1] == 2:
                sample = sample[:, 0]
            cpu_outputs[f"sample_{i}"] = sample

    for key, val in copied_outputs.items():
        if isinstance(val, Tensor):
            cpu_outputs[key] = _writer_array(val)
        else:
            cpu_outputs[key] = _writer_array(val)

    df = pd.DataFrame.from_dict(cpu_outputs)

    # check if path already exists, then just append
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:  # create new csv
        df.to_csv(path, index=False)


def save_classification_predictions(
    outputs: STEP_OUTPUT,
    path: str,
    task: str = "multiclass",
    binary_encoding: str = "two_logit",
) -> None:
    """Save classification predictions to csv file.

    .. versionchanged:: 0.4.0

       The writer is non-mutating, handles a one-logit binary task explicitly,
       and rank-shards distributed output to avoid append collisions.

    For the "multilabel" task the labels are not mutually exclusive, so instead
    of a single ``pred`` column there is one ``pred_i`` column per label,
    holding the thresholded prediction for that label.

    Args:
        outputs: metrics and values to be saved
            - logits: logits of shape [batch_size, num_classes]
            - pred: predictions of shape [batch_size, num_classes]
            - target: targets of shape [batch_size], or [batch_size, num_labels]
              for the "multilabel" task
            - pred_uct: predictive uncertainty of shape [batch_size]
        path: path where csv should be saved
        task: one of "binary", "multiclass" or "multilabel"
        binary_encoding: binary probability encoding for the one-logit CSV
            thresholding case.
    """
    # Lightning types the hook argument as STEP_OUTPUT; the UQ methods always
    # return a dict from test_step, so narrow once here rather than at every hook.
    assert isinstance(outputs, dict)
    copied_outputs = dict(outputs)
    path = _distributed_path(path)
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if "samples" in copied_outputs:
        _ = copied_outputs.pop("samples")
    if "logits" in copied_outputs:
        _ = copied_outputs.pop("logits")

    pred_set_true = "pred_set" in copied_outputs

    if pred_set_true:
        pred_set = [
            str(tensor.cpu().numpy().tolist())
            for tensor in copied_outputs.pop("pred_set")
        ]
        df_pred_set = pd.DataFrame(pred_set, columns=["pred_set"])

    # save inidividual predictions as class probs
    class_probs = copied_outputs.pop("pred")
    if task == "binary" and binary_encoding == "one_logit" and class_probs.ndim == 1:
        class_probs = class_probs.unsqueeze(1)

    for i in range(class_probs.shape[1]):
        copied_outputs[f"class_prob_{i}"] = class_probs[:, i]

    cpu_outputs = {}
    for key, val in copied_outputs.items():
        val = _writer_array(val)
        # per-label targets and the like need one column each
        if val.ndim == 2:
            for i in range(val.shape[1]):
                cpu_outputs[f"{key}_{i}"] = val[:, i]
        else:
            cpu_outputs[key] = val

    if task == "multilabel":
        preds = (class_probs > 0.5).int().cpu().numpy()
        df_pred = pd.DataFrame(
            preds, columns=[f"pred_{i}" for i in range(preds.shape[1])]
        )
    elif task == "binary" and binary_encoding == "one_logit":
        pred_class = (class_probs[:, 0] > 0.5).int().cpu().numpy()
        df_pred = pd.DataFrame(pred_class, columns=["pred"])
    else:
        pred_class = torch.argmax(class_probs, dim=1).cpu().numpy()
        df_pred = pd.DataFrame(pred_class, columns=["pred"])

    # Create DataFrame for the rest of the outputs
    df_outputs = pd.DataFrame.from_dict(cpu_outputs)

    # Concatenate the two DataFrames
    df = pd.concat([df_pred, df_outputs], axis=1)

    if pred_set_true:
        df = pd.concat([df, df_pred_set], axis=1)

    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def map_stochastic_modules(
    model: nn.Module, stochastic_module_names: None | list[str, int]
) -> list[str]:
    """Retrieve desired stochastic module names from user arg.

    Args:
        model: model from which to retrieve the module names
        stochastic_module_names: argument to uq_method for partial stochasticity

    Returns:
        list of desired partially stochastic module names
    """
    ordered_module_names: list[str] = []
    # ignore batchnorm
    for name, val in model.named_parameters():
        # module = getattr(model, )
        ordered_module_names.append(".".join(name.split(".")[:-1]))
    ordered_module_names = list(OrderedDict.fromkeys(ordered_module_names))

    # split of weight/bias
    ordered_module_params = [
        name for name, val in list(model.named_parameters())
    ]  # all
    module_names = [".".join(name.split(".")[:-1]) for name in ordered_module_params]
    # remove duplicates due to weight/bias
    module_names = list(set(module_names))

    module_names = [name for name in module_names if name != ""]  # remove empty string

    if not stochastic_module_names:  # None means fully stochastic
        part_stoch_names = module_names.copy()
    elif all(isinstance(elem, int) for elem in stochastic_module_names):
        part_stoch_names = [
            ordered_module_names[idx] for idx in stochastic_module_names
        ]  # retrieve last ones
    elif all(isinstance(elem, str) for elem in stochastic_module_names):
        assert set(stochastic_module_names).issubset(module_names), (
            f"Model only contains these parameter modules {module_names}, "
            f"and you requested {stochastic_module_names}."
        )
        part_stoch_names = stochastic_module_names
    else:
        raise ValueError
    return part_stoch_names


def _get_input_layer_name_and_module(model: nn.Module) -> tuple[str, nn.Module]:
    """Retrieve the input layer name and module from a pytorch model.

    Args:
        model: pytorch model

    Returns:
        input key and module
    """
    keys = []
    children = list(model.named_children())
    while children != []:
        name, module = children[0]
        keys.append(name)
        children = list(module.named_children())

    key = ".".join(keys)
    return key, module


def _get_output_layer_name_and_module(model: nn.Module) -> tuple[str, nn.Module]:
    """Retrieve the output layer name and module from a pytorch model.

    Args:
        model: pytorch model

    Returns:
        output key and module
    """
    queue = list(model.named_modules())
    last_module_with_out = None
    last_keys_with_out = None

    while queue:
        name, module = queue.pop(0)
        if hasattr(module, "out_features") or hasattr(module, "out_channels"):
            last_module_with_out = module
            last_keys_with_out = name

    if last_module_with_out is None:
        raise ValueError("No layer with out_features found.")

    return last_keys_with_out, last_module_with_out


def _get_num_inputs(model: nn.Module) -> int:
    """Get the number of inputs for a module.

    Args:
        model: pytorch model

    Returns:
        number of inputs to the model
    """
    _, module = _get_input_layer_name_and_module(model)
    if hasattr(module, "in_features"):  # Linear Layer
        num_inputs = module.in_features
    elif hasattr(module, "in_channels"):  # Conv Layer
        num_inputs = module.in_channels
    else:
        raise ValueError(f"Module {module} does not have in_features or in_channels.")
    return num_inputs


def _get_num_outputs(model: nn.Module) -> int:
    """Get the number of outputs for a module.

    Args:
        model: pytorch model

    Returns:
        number of outputs from the model
    """
    _, module = _get_output_layer_name_and_module(model)
    if hasattr(module, "out_features"):  # Linear Layer
        num_outputs = module.out_features
    elif hasattr(module, "out_channels"):  # Conv Layer
        num_outputs = module.out_channels
    else:
        raise ValueError(f"Module {module} does not have out_features or out_channels.")
    return num_outputs


def freeze_model_backbone(model: nn.Module) -> None:
    """Freeze the backbone of a model.

    Args:
        model: pytorch model
    """
    for param in model.parameters():
        param.requires_grad = False

    # for timm model
    get_classifier = getattr(model, "get_classifier", None)
    if callable(get_classifier):
        for param in get_classifier().parameters():
            param.requires_grad = True
    else:
        # find last layer
        _, module = _get_output_layer_name_and_module(model)
        for param in module.parameters():
            param.requires_grad = True


def freeze_segmentation_model(
    model: nn.Module, freeze_backbone: bool, freeze_decoder: bool
) -> None:
    """Freeze the encoder or decoder of a segmentation model.

    Args:
        model: pytorch model
        freeze_backbone: whether to freeze the model backbone
        freeze_decoder: whether to freeze the decoder
    """
    # Freeze backbone
    if hasattr(model, "encoder") and freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Freeze decoder
    if hasattr(model, "decoder") and freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by name.

    Args:
        model: full model
        module_name: name of module to replace within model
        new_module: initialized module which is the replacement
    """
    module_levels = module_name.split(".")
    last_level = module_levels[-1]
    if len(module_levels) == 1:
        setattr(model, last_level, new_module)
    else:
        setattr(getattr(model, ".".join(module_levels[:-1])), last_level, new_module)
