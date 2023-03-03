"""Utility functions for Evaluation."""

import glob
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
import uncertainty_toolbox as uct
from tqdm import tqdm

from .utils import read_config


def compute_results_all_experiments(exp_collection_dir: str, save_path: str) -> None:
    """Compute the results over a collection of experiments.

    Args:
        exp_collection_dir: directory where experiments are collected
        save_path: path where to save the results
    """
    exp_dirs = glob.glob(os.path.join(exp_collection_dir, "*"), recursive=True)
    # remove the results directory
    exp_dirs.remove("./experiments/experiments/exp_results")

    exp_dfs = []
    for exp_dir in tqdm(exp_dirs):
        exp_dfs.append(compute_results_over_seeds(exp_dir))

    all_exp_df = pd.concat(exp_dfs, ignore_index=True)
    all_exp_df.to_csv(save_path, index=False)


def compute_results_over_seeds(exp_dir: str) -> pd.DataFrame:
    """Compute the results over seeded experiments."""
    seed_dirs = glob.glob(os.path.join(exp_dir, "*"), recursive=True)
    # remove experiment level .yaml file
    seed_dirs = [dir for dir in seed_dirs if not dir.endswith(".yaml")]

    seed_dfs = []
    for seed_dir in seed_dirs:
        df, config_dict = compute_metrics_for_single_seed(seed_dir)
        seed_dfs.append(df)

    # compute mean across seeds
    seed_df = pd.concat(seed_dfs, ignore_index=True)
    # seed_df = pd.DataFrame(seed_df.mean(0)).T

    # add meta data
    for key, val in config_dict.items():
        seed_df[key] = val

    seed_df["seed"] = range(len(seed_df))

    return seed_df


def compute_metrics_for_single_seed(
    seed_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute the metrics for a single seed."""
    pred_dir = os.path.join(seed_dir, "prediction")
    pred_csv = os.path.join(pred_dir, "predictions.csv")

    pred_df = pd.read_csv(pred_csv)

    uq_metrics = uct.metrics.get_all_metrics(
        pred_df["mean"].values.squeeze(),
        pred_df["pred_uct"].values.squeeze(),
        pred_df["targets"].values.squeeze(),
        verbose=False,
    )

    uq_metric_categories = ["scoring_rule", "avg_calibration", "sharpness", "accuracy"]
    metrics_dict = {uq_cat: uq_metrics[uq_cat] for uq_cat in uq_metric_categories}

    # mulit column df holding the results
    df = pd.DataFrame.from_dict(metrics_dict, orient="index").stack().to_frame().T

    # drop multilevel
    df.columns = df.columns.droplevel(0)

    # now add information about this experiment from the config_file
    config_path = os.path.join(pred_dir, "seed_config.yaml")
    seed_config = read_config(config_path)

    # extract date
    date = "_".join(os.path.basename(os.path.dirname(seed_dir)).split("_")[-2:])
    date_time = datetime.strptime(date, "%m-%d-%Y_%H:%M:%S")

    config_dict = {
        "base_model": seed_config["model"]["base_model"],
        "loss_fn": seed_config["model"]["loss_fn"],
        "ensemble": seed_config["model"].get("ensemble", None),
        "ensemble_members": seed_config["model"].get("ensemble_members", 1),
        "conformalized": seed_config["model"].get("conformalized", False),
        "dataset_name": seed_config["ds"]["dataset_name"],
        "pred_log_dir": seed_config["experiment"]["save_dir"],
        "mlp_n_outputs": seed_config["model"]["mlp"]["n_outputs"],
        "date": date_time,
    }

    return df, config_dict
