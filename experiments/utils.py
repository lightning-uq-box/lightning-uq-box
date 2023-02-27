"""Utility functions."""

from typing import Any, Dict

from ruamel.yaml import YAML


def read_config(config_path: str) -> Dict[str, Any]:
    """Open config file."""
    yaml = YAML()
    with open(config_path) as fd:
        config = yaml.load(fd)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save config file."""
    yaml = YAML()
    with open(path, "w") as fd:
        yaml.dump(config, fd)
