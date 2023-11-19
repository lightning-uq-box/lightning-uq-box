"""Command-line interface to TorchGeo."""


from lightning.pytorch.cli import ArgsType, LightningCLI

# Allows classes to be referenced using only the class name
import lightning_uq_box.datamodules  # noqa: F401
import lightning_uq_box.uq_methods  # noqa: F401


def main(args: ArgsType = None) -> None:
    """Command-line interface to Lightning-UQ-Box."""
    LightningCLI(
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=args,
    )
