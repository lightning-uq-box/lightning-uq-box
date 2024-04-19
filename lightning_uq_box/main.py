# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Command-line interface to Lightning-UQ-Box."""

from lightning.pytorch.cli import ArgsType, LightningCLI


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
    get_uq_box_cli(args)
