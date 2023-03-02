"""Compute Results for UCI experiments."""

import argparse

from experiments.eval_utils import compute_results_all_experiments


def collect_results(exp_collection_dir: str):
    """Collect results for UCI experiments.

    Args:
        exp_collection_dir: path pointing to root dir of experiments
    """
    compute_results_all_experiments(exp_collection_dir)


def start() -> None:
    """Collect results for UCI Experiment."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="compute_results_uci.py",
        description="Runs an experiment for a given config file.",
    )

    parser.add_argument(
        "--exp_dir_path", help="Path to the root of experiments", required=True
    )

    args = parser.parse_args()

    collect_results(args.exp_dir_path)


if __name__ == "__main__":
    start()
