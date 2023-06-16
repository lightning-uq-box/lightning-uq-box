"""Utility functions."""
from typing import Any


def ignore_args(dictionary: dict[str, Any], ignore_keys: list[str]) -> dict[str, Any]:
    """Ignore certain arguments in dictionary.

    Args:
        dictionary:
        ignore_keys: which keys to ignore

    Returns:
        dictionary with ingor keys removed
    """
    return {arg: val for arg, val in dictionary.items() if arg not in ignore_args}
