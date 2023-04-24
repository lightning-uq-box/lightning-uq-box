"""Utility functions."""

import os
from datetime import datetime
from typing import Any

from ruamel.yaml import YAML


def ignore_args(dictionary: dict[str, Any], ignore_keys: list[str]) -> dict[str, Any]:
    """Ignore certain arguments in dictionary.

    Args:
        dictionary:
        ignore_keys: which keys to ignore

    Returns:
        dictionary with ingor keys removed
    """
    return {arg: val for arg, val in dictionary.items() if arg not in ignore_args}


def create_experiment_dir(config: dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['model']['base_model']}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    return config


def read_config(config_path: str) -> dict[str, Any]:
    """Open config file."""
    yaml = YAML()
    with open(config_path) as fd:
        config = yaml.load(fd)
    return config


def save_config(config: dict[str, Any], path: str) -> None:
    """Save config file."""
    yaml = YAML()
    with open(path, "w") as fd:
        yaml.dump(config, fd)
