# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Command-line interface to Lightning-UQ-Box."""

from lightning.pytorch.cli import ArgsType, LightningCLI

# Allows classes to be referenced using only the class name
import lightning_uq_box.datamodules  # noqa: F401
import lightning_uq_box.uq_methods  # noqa: F401
from lightning_uq_box.models import MLP  # noqa: F401


def get_uq_box_cli(args: ArgsType = None) -> LightningCLI:
    """Get Command-line interface Object for Lightning-UQ-Box."""
    return LightningCLI(
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
        run=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        args=args,
    )


def main(args: ArgsType = None) -> None:
    """Command-line interface to Lightning-UQ-Box."""
    LightningCLI(
        # model_class=MCDropoutRegression,
        # datamodule_class=ToyHeteroscedasticDatamodule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        args=args,
    )
