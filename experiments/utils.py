"""Utility functions."""

import os
from datetime import datetime
from typing import Any, Dict, List


def ignore_args(dictionary: Dict[str, Any], ignore_keys: List[str]) -> Dict[str, Any]:
    """Ignore certain arguments in dictionary.

    Args:
        dictionary:
        ignore_keys: which keys to ignore

    Returns:
        dictionary with ingor keys removed
    """
    return {arg: val for arg, val in dictionary.items() if arg not in ignore_args}


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['uq_method']['_target_'].split('.')[-1]}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    return config
