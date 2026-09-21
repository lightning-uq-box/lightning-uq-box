# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Small shared helpers for 0.4 compatibility adapters.

Deprecated classes intentionally bind their own exports in their historical
modules.  Keeping the warning helper here avoids a replacement class factory,
which would obscure signatures and can change checkpoint state-dict prefixes.
"""

from warnings import warn


def warn_legacy_adapter(old_name: str, replacement: str) -> None:
    """Warn callers that a concrete task class is a 0.4 compatibility adapter."""
    warn(
        f"{old_name} is deprecated and will be removed in 0.5; use {replacement} "
        "with an explicit task value instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = ["warn_legacy_adapter"]
