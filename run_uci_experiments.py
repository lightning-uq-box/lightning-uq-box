"""Reproduce UCI Regression Dataset results."""

import argparse
import os

from experiments.setup_experiment import generate_model, generate_trainer
from experiments.utils import create_experiment_dir, read_config, save_config
from uq_method_box.datamodules import UCIRegressionDatamodule
from uq_method_box.models import MLP


def run(config_path: str) -> None:
    """Run the UCI experiment.

    Args:
        config_path: path to config file
    """
    config = read_config(config_path)
    config = create_experiment_dir(config)

    # generate datamodule
    dm = UCIRegressionDatamodule(config)

    # get the number of features to update the number of inputs to model
    config["model"]["mlp"]["n_inputs"] = dm.uci_ds.num_features

    # generate mlp
    mlp = MLP(**config["model"]["mlp"])

    # generate model
    model = generate_model(config, mlp)

    # generate trainer
    trainer = generate_trainer(config)

    # fit model
    trainer.fit(model, dm)

    # make predictions on test set
    trainer.test(model, dataloaders=dm.test_dataloader())

    # save config to experiment directory
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))

    print("Successfully completed experiment run.")


def start() -> None:
    """Start UCI Experiment."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Runs an experiment for a given config file.",
    )

    parser.add_argument("--config_path", help="Path to the config file", required=True)

    args = parser.parse_args()

    run(args.config_path)


if __name__ == "__main__":
    start()
