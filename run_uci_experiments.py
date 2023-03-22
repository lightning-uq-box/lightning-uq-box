"""Reproduce UCI Regression Dataset results."""

import argparse
import copy
import glob
import os

from experiments.setup_experiment import (
    generate_base_model,
    generate_ensemble_model,
    generate_trainer,
)
from experiments.utils import create_experiment_dir, read_config, save_config
from uq_method_box.datamodules import UCIRegressionDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import CQR, LaplaceModel


def run(config_path: str) -> None:
    """Run the UCI experiment.

    Args:
        config_path: path to config file
    """
    config = read_config(config_path)

    # main directory for the
    experiment_config = create_experiment_dir(config)

    for i in range(3):  # 20 random seeds
        seed_config = copy.deepcopy(experiment_config)

        # create a subdirectory for this seed
        seed_dir = os.path.join(seed_config["experiment"]["save_dir"], f"seed_{i}")
        os.makedirs(seed_dir)

        seed_config["experiment"]["save_dir"] = seed_dir

        seed_config["ds"]["seed"] = i

        # if no ensembling this will just execute once
        for m in range(seed_config["model"].get("ensemble_members", 1)):
            run_config = copy.deepcopy(seed_config)

            # create subdirectory for each run within a seed and overwrite run_config
            model_run_dir = os.path.join(seed_dir, f"model_{m}")
            os.makedirs(model_run_dir)

            run_config["experiment"]["save_dir"] = model_run_dir

            dm = UCIRegressionDatamodule(config)

            # get the number of features to update the number of inputs to model
            run_config["model"]["model_args"]["n_inputs"] = dm.uci_ds.num_features

            model = generate_base_model(run_config, model_class=MLP)

            # generate trainer
            trainer = generate_trainer(run_config)

            # fit model
            trainer.fit(model, dm)

            # save config for this particular run
            save_config(
                run_config,
                os.path.join(run_config["experiment"]["save_dir"], "run_config.yaml"),
            )

        # new directory in seed directory where we save the prediction results
        prediction_config = copy.deepcopy(seed_config)
        pred_dir = os.path.join(
            prediction_config["experiment"]["save_dir"], "prediction"
        )
        prediction_config["experiment"]["save_dir"] = pred_dir
        os.makedirs(pred_dir)

        # Laplace Model Wrapper
        if "laplace" in run_config["model"]:
            # laplace requires train data loader post fit, maybe there is also
            # a more elegant way to solve this
            model = LaplaceModel(
                model,
                prediction_config["model"]["laplace_args"],
                dm.train_dataloader(),
                prediction_config["experiment"]["save_dir"],
            )

        # if we want to build an ensemble
        if seed_config["model"].get("ensemble_members", 1) >= 2:
            # make predictions with the deep ensemble wrapper
            # load all checkpoints into instantiated models
            ensemble_ckpt_paths = glob.glob(
                os.path.join(seed_config["experiment"]["save_dir"], "**", "*.ckpt"),
                recursive=True,
            )
            ensemble_members = []
            for ckpt_path in ensemble_ckpt_paths:
                ensemble_members.append(
                    {"model_class": type(model), "ckpt_path": ckpt_path}
                )

            model = generate_ensemble_model(prediction_config, ensemble_members)

        # conformal prediction step if requested should happen here
        if seed_config["model"].get("conformalized", False):
            # wrap model in CQR
            model = CQR(
                model,
                seed_config["model"]["quantiles"],
                dm.calibration_dataloader(),
                seed_config["experiment"]["save_dir"],
            )

        # generate trainer for test to save in prediction dir
        trainer = generate_trainer(prediction_config)

        # and update save dir in model config to save it separately
        model.hparams.save_dir = pred_dir

        # make predictions on test set, check that it still takes the best model?
        trainer.test(model, dataloaders=dm.test_dataloader())

        # save run_config to sub directory for the seed experiment directory
        save_config(seed_config, os.path.join(pred_dir, "seed_config.yaml"))

    # save the config to the upper experiment directory
    save_config(
        config,
        os.path.join(
            experiment_config["experiment"]["save_dir"], "experiment_config.yaml"
        ),
    )

    print("finished experiments for all seeds")


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
