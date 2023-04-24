"""Experiment generator to setup an experiment based on a config file."""

from typing import Any, Union

import torch.nn as nn
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from uq_method_box.uq_methods import (
    BaseModel,
    BayesByBackpropModel,
    DeepEnsembleModel,
    DERModel,
    DeterministicGaussianModel,
    MCDropoutModel,
    QuantileRegressionModel,
    SGLDModel,
)
from uq_method_box.uq_methods.utils import retrieve_optimizer


def generate_base_model(
    config: dict[str, Any], model_class: type[nn.Module]
) -> LightningModule:
    """Generate a configured base model.

    Args:
        config: config dictionary

    Keywoard Args:
        model_class: optional to initiate a base class with a certain model
        train_loader: needed for laplace model

    Returns:
        configured pytorch lightning module
    """
    if config["model"]["base_model"] == "base_model":
        return BaseModel(
            model_class,
            model_args=config["model"]["model_args"],
            optimizer=retrieve_optimizer(config["optimizer"]),
            optimizer_args=config["optimizer_args"],
            lr=config["optimizer"]["lr"],
            loss_fn=config["model"]["loss_fn"],
            save_dir=config["experiment"]["save_dir"],
        )

    elif config["model"]["base_model"] == "mc_dropout":
        return MCDropoutModel(
            model_class,
            model_args=config["model"]["model_args"],
            num_mc_samples=config["model"]["num_mc_samples"],
            lr=config["optimizer"]["lr"],
            loss_fn=config["model"]["loss_fn"],
            save_dir=config["experiment"]["save_dir"],
            burnin_epochs=config["model"]["burnin_epochs"],
            max_epochs=config["pl"]["max_epochs"],
        )

    elif config["model"]["base_model"] == "quantile_regression":
        return QuantileRegressionModel(
            model_class,
            model_args=config["model"]["model_args"],
            lr=config["optimizer"]["lr"],
            save_dir=config["experiment"]["save_dir"],
            quantiles=config["model"]["quantiles"],
        )

    elif config["model"]["base_model"] == "gaussian":
        return DeterministicGaussianModel(
            model_class,
            model_args=config["model"]["model_args"],
            optimizer=retrieve_optimizer(config["optimizer"]),
            optimizer_args=config["optimizer_args"],
            loss_fn=config["model"]["loss_fn"],
            save_dir=config["experiment"]["save_dir"],
            burnin_epochs=config["model"]["burnin_epochs"],
            max_epochs=config["pl"]["max_epochs"],
        )

    elif config["model"]["base_model"] == "sgld":
        return SGLDModel(
            model_class,
            model_args=config["model"]["model_args"],
            n_sgld_samples=config["model"]["n_sgld_samples"],
            burnin_epochs=config["model"]["burnin_epochs"],
            loss_fn=config["model"]["loss_fn"],
            save_dir=config["experiment"]["save_dir"],
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
            noise_factor=config["optimizer"]["noise_factor"],
        )

    elif config["model"]["base_model"] == "bayes_by_backprop":
        return BayesByBackpropModel(
            model_class,
            model_args=config["model"]["model_args"],
            lr=config["optimizer"]["lr"],
            loss_fn=config["model"]["loss_fn"],
            save_dir=config["experiment"]["save_dir"],
            **config["model"]["bayes_by_backprop"],
        )

    elif config["model"]["base_model"] == "der":
        return DERModel(
            model_class,
            model_args=config["model"]["model_args"],
            lr=config["optimizer"]["lr"],
            save_dir=config["experiment"]["save_dir"],
        )
    else:
        raise ValueError("Your base_model choice is currently not supported.")


def generate_ensemble_model(
    config: dict[str, Any],
    ensemble_members: list[dict[str, Union[type[LightningModule], str]]],
) -> LightningModule:
    """Generate an ensemble model.

    Args:
        config: config dictionary
        ensemble_members: ensemble models

    Returns:
        configureed ensemble lightning module
    """
    if config["model"]["ensemble"] == "deep_ensemble":
        return DeepEnsembleModel(
            ensemble_members,
            config["experiment"]["save_dir"],
            config["model"]["quantiles"],
        )

    # multi swag

    # mc-dropout ensemble similar to swag

    else:
        raise ValueError("Your ensemble choice is currently not supported.")


def generate_datamodule(config: dict[str, Any]) -> LightningDataModule:
    """Generate LightningDataModule from config file.

    Args:
        config: config dictionary

    Returns:
        configured datamodule for experiment
    """
    pass


def generate_trainer(config: dict[str, Any]) -> Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [
        CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
        # WandbLogger(
        #   save_dir=config["experiment"]["save_dir"],
        #    project=config["wandb"]["project"],
        #    entity=config["wandb"]["entity"],
        #    resume="allow",
        #    config=config,
        #    mode=config["wandb"].get("mode", "online"),
        # ),
    ]

    track_metric = "train_loss"
    mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"],
        save_top_k=1,
        monitor=track_metric,
        mode=mode,
        every_n_epochs=1,
    )

    return Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=[checkpoint_callback],
        logger=loggers,
    )
