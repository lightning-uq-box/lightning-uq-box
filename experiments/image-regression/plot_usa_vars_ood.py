import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from tqdm import tqdm

plt.style.use("fivethirtyeight")


def compute_metrics_for_predictions(csv_path: str) -> pd.DataFrame:
    """Compute metrics for prediction file.

    Args:
        save_dir: path_to_csv

    Returns:
        dataframe with computed metrics
    """
    pred_df = pd.read_csv(csv_path)

    uq_metrics = uct.metrics.get_all_metrics(
        pred_df["pred"].values.squeeze(),
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
    df["split"] = csv_path.split(os.sep)[-1].split(".")[0]

    return df


def create_plot(full_df: pd.DataFrame) -> plt.Figure:
    """Create plot for levels of OOD.

    Args:
        full_df: df that holds all metric information

    Returns:
        figure
    """
    models = full_df["model_name"].unique()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for model in models:
        model_df = full_df[full_df["model_name"] == model]
        model_df.sort_values(by="split", inplace=True)

        ax[0, 0].plot(model_df["split"], model_df["rmse"], label=model)
        ax[0, 0].set_title("RMSE")
        ax[0, 0].set_xticks([])
        ax[0, 1].plot(model_df["split"], model_df["mae"], label=model)
        ax[0, 1].set_title("MAE")
        ax[0, 1].set_xticks([])

        ax[1, 0].plot(model_df["split"], model_df["nll"], label=model)
        ax[1, 0].set_title("NLL")
        ax[1, 0].tick_params("x", labelrotation=60)
        ax[1, 1].plot(model_df["split"], model_df["sharp"], label=model)
        ax[1, 1].set_title("Sharpness")
        ax[1, 1].tick_params("x", labelrotation=60)

    plt.legend()
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    exp_dirs = [
        "/mnt/SSD2/nils/uq-method-box/experiments/image-regression/experiments/usa_vars/usa_vars_BaseModel_LaplaceModel_06-19-2023_09-51-43",
        # "/mnt/SSD2/nils/uq-method-box/experiments/image-regression/experiments/usa_vars/usa_vars_QuantileRegressionModel_06-17-2023_14-00-04",
        "/mnt/SSD2/nils/uq-method-box/experiments/image-regression/experiments/usa_vars/usa_vars_MCDropoutModel_06-17-2023_13-59-51",
    ]
    full_df = pd.DataFrame()
    for exp_dir in exp_dirs:
        pred_csvs = glob.glob(os.path.join(exp_dir, "pred*.csv"))
        model_id = exp_dir.split(os.sep)[-1].split("_")[-3]
        for csv_path in tqdm(pred_csvs):
            uct_df = compute_metrics_for_predictions(csv_path)
            uct_df["model_name"] = model_id
            full_df = pd.concat([full_df, uct_df])

    fig = create_plot(full_df)
    fig.savefig("usa_vars_ood.png")

    import pdb

    pdb.set_trace()
    print(0)
