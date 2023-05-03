"""Experiment generator to setup an experiment based on a config file."""

from typing import Any, Dict, List, Union

from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from uq_method_box.uq_methods import DeepEnsembleModel


def generate_ensemble_model(
    config: Dict[str, Any],
    ensemble_members: List[Dict[str, Union[type[LightningModule], str]]],
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


def generate_datamodule(config: Dict[str, Any]) -> LightningDataModule:
    """Generate LightningDataModule from config file.

    Args:
        config: config dictionary

    Returns:
        configured datamodule for experiment
    """
    pass


def generate_trainer(config: Dict[str, Any]) -> Trainer:
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

    return instantiate(
        config.trainer,
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=[checkpoint_callback],
        logger=loggers,
    )
