"""Command-line interface to TorchGeo."""


from jsonargparse import lazy_instance
from lightning.pytorch.cli import ArgsType, LightningCLI

# Allows classes to be referenced using only the class name
import lightning_uq_box.datamodules  # noqa: F401
import lightning_uq_box.uq_methods  # noqa: F401
from lightning_uq_box.models import MLP

# class MyLightningCLI(LightningCLI):
# def add_arguments_to_parser(self, parser):
#     import pdb
#     pdb.set_trace()
#     parser.set_defaults({"model.backbone": lazy_instance(MLP, encoder_layers=24)})


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


# TODO
# this should solve the optimizer problem
# https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#multiple-optimizers-and-schedulers
