"""Run experiments for the Paper."""

import argparse
import os

from experiments.setup_experiment import (
    generate_datamodule,
    generate_model,
    generate_trainer,
)
from experiments.utils import create_experiment_dir, read_config, save_config

# chang here

def run(config_path: str):
    """Training and Evaluation Script.

    Args:
        config_path: path to config file
    """
    config = read_config(config_path)
    config = create_experiment_dir(config)

    # generate model
    model = generate_model(config)

    datamodule = generate_datamodule(config)

    # updated config after datamodule
    config = datamodule.config

    # generate trainer
    trainer = generate_trainer(config)
    trainer.log_every_n_steps = min(  # type: ignore[attr-defined]
        len(datamodule.train_dataloader()), config["pl"]["log_every_n_steps"]
    )
    print(f"Training data loader length {len(datamodule.train_dataloader())}.")
    print(f"Using GPU device {trainer.device_ids}")

    # fit model
    trainer.fit(model, datamodule)

    # possible "postprocessing" steps like Laplace or CQR

    # prediction phase

    # save config
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))

    print("Successfully completed experiment run")


def start():
    """Start training."""
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
