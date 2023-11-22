"""Command-line interface to Lightning-UQ-Box."""

from lightning.pytorch.cli import ArgsType, LightningCLI

# Allows classes to be referenced using only the class name
import lightning_uq_box.datamodules  # noqa: F401
import lightning_uq_box.uq_methods  # noqa: F401
from lightning_uq_box.models import MLP


def get_uq_box_cli(args: ArgsType = None) -> LightningCLI:
    """Get Command-line interface Object for Lightning-UQ-Box."""
    return LightningCLI(
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
        run=False,
        args=args,
    )


def main(args: ArgsType = None) -> None:
    """Command-line interface to Lightning-UQ-Box."""
    LightningCLI(
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        auto_configure_optimizers=False,
        args=args,
    )
