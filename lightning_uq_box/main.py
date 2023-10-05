"""Command-line interface to Lightning-UQ-Box."""

from lightning.pytorch.cli import ArgsType, LightningCLI
import lightning as L

# Allows classes to be referenced using only the class name
import lightning_uq_box.datamodules  # noqa: F401
import lightning_uq_box.models  # noqa: F401
import lightning_uq_box.uq_methods  # noqa: F401
from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import BaseModule


class UQBoxLightningCLI(LightningCLI):
    subclass_mode_model = True
    auto_configure_optimizers = False


def main(args: ArgsType = None) -> None:
    """Command-line interface to Lightning-UQ-Box."""

    LightningCLI(
        model_class=L.LightningModule,
        datamodule_class=L.LightningDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
        args=args,
    )
