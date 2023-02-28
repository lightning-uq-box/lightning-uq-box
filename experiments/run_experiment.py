"""Run experiments for the Paper."""

import argparse
import os
from datetime import datetime
from typing import Any, Dict

from experiment_generator import generate_datamodule, generate_model, generate_trainer
from utils import read_config, save_config


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['model']['model']}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    return config


def run(config_path):
    """Training and Evaluation Script."""

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
