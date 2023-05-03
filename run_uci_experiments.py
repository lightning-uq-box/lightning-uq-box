"""Reproduce UCI Regression Dataset results."""

import argparse
import copy
import glob
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from experiments.setup_experiment import generate_trainer
from experiments.utils import create_experiment_dir
from uq_method_box.uq_methods import CQR


def run(config_path: str) -> None:
    """Run the UCI experiment.

    Args:
        config_path: path to config file
    """
    conf = OmegaConf.load(config_path)

    # main directory for the
    experiment_config = create_experiment_dir(conf)

    for i in range(2):  # 20 random seeds
        seed_config = copy.deepcopy(experiment_config)

        # create a subdirectory for this seed
        seed_dir = os.path.join(seed_config["experiment"]["save_dir"], f"seed_{i}")
        os.makedirs(seed_dir)

        seed_config["experiment"]["save_dir"] = seed_dir

        seed_config["datamodule"]["seed"] = i

        # if no ensembling this will just execute once
        if "post_processing" in seed_config:
            num_ensembles = seed_config["post_processing"].get("n_ensemble_members", 1)
        else:
            num_ensembles = 1

        for m in range(num_ensembles):
            run_config = copy.deepcopy(seed_config)

            # create subdirectory for each run within a seed and overwrite run_config
            model_run_dir = os.path.join(seed_dir, f"model_{m}")
            os.makedirs(model_run_dir)

            run_config["experiment"]["save_dir"] = model_run_dir
            run_config["uq_method"]["save_dir"] = model_run_dir

            dm = instantiate(run_config.datamodule)

            # get the number of features to update the number of inputs to model
            try:
                run_config["uq_method"]["model"]["n_inputs"] = dm.uci_ds.num_features
            except ConfigKeyError:  # DKL model
                run_config["uq_method"]["feature_extractor"][
                    "n_inputs"
                ] = dm.uci_ds.num_features

            model = instantiate(run_config.uq_method)

            # generate trainer
            trainer = generate_trainer(run_config)

            # fit model
            trainer.fit(model, dm)

            # save config for this particular run
            with open(
                os.path.join(
                    experiment_config["experiment"]["save_dir"], "run_config.yaml"
                ),
                "w",
            ) as fp:
                OmegaConf.save(config=run_config, f=fp.name)

        # new directory in seed directory where we save the prediction results
        prediction_config = copy.deepcopy(seed_config)
        pred_dir = os.path.join(
            prediction_config["experiment"]["save_dir"], "prediction"
        )
        prediction_config["experiment"]["save_dir"] = pred_dir
        os.makedirs(pred_dir)

        # Laplace Model Wrapper
        if "post_processing" in run_config:
            if "LaplaceModel" in run_config["post_processing"]["_target_"]:
                # laplace requires train data loader post fit, maybe there is also
                # a more elegant way to solve this
                model = instantiate(
                    run_config.post_processing,
                    model=model.model,
                    train_loader=dm.train_dataloader(),
                    save_dir=run_config["experiment"]["save_dir"],
                )

            # SWAG Model Wrapper
            if "SWAGModel" in run_config["post_processing"]["_target_"]:
                # swag requires train data loader post fit, maybe also more elegant way
                # to solve this
                model = instantiate(
                    run_config.post_processing,
                    model=model.model,
                    train_loader=dm.train_dataloader(),
                    save_dir=run_config["experiment"]["save_dir"],
                )

            # if we want to build an ensemble
            if "DeepEnsemble" in run_config["post_processing"]["_target_"]:
                # make predictions with the deep ensemble wrapper
                # load all checkpoints into instantiated models
                ensemble_ckpt_paths = glob.glob(
                    os.path.join(seed_config["experiment"]["save_dir"], "**", "*.ckpt"),
                    recursive=True,
                )
                ensemble_members = []
                for ckpt_path in ensemble_ckpt_paths:
                    ensemble_members.append(
                        {
                            # "model_class": type(model),
                            "ckpt_path": ckpt_path,
                            "base_model": run_config.uq_method,
                        }
                    )

                model = instantiate(
                    prediction_config.post_processing,
                    ensemble_members=ensemble_members,
                    save_dir=run_config["experiment"]["save_dir"],
                    _convert_="object",
                )

        # conformal prediction step if requested should happen here
        if "calibration" in seed_config:
            # wrap model in CQR
            model = CQR(
                model,
                model.quantiles,
                dm.calibration_dataloader(),
                seed_config.experiment["save_dir"],
            )

        # generate trainer for test to save in prediction dir
        trainer = instantiate(prediction_config.trainer)

        # and update save dir in model config to save it separately
        model.hparams.save_dir = pred_dir

        # make predictions on test set, check that it still takes the best model?
        trainer.test(model, dataloaders=dm.test_dataloader())

        # save run_config to sub directory for the seed experiment directory
        with open(os.path.join(pred_dir, "seed_config.yaml"), "w") as fp:
            OmegaConf.save(config=seed_config, f=fp.name)

    # save the config to the upper experiment directory
    with open(
        os.path.join(
            experiment_config["experiment"]["save_dir"], "experiment_config.yaml"
        ),
        "w",
    ) as fp:
        OmegaConf.save(config=conf, f=fp.name)

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
