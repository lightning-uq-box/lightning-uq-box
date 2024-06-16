# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Visualization utils for Lightning-UQ-Box."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct


def plot_training_metrics(save_dir: str, metrics: list[str]) -> plt.figure:
    """Plot training metrics from latest lightning CSVLogger version.

    Args:
        save_dir: path to save directory of CSVLogger
        metrics: list of metrics to plot
    """
    latest_version = sorted(os.listdir(save_dir))[-1]
    metrics_path = os.path.join(save_dir, latest_version, "metrics.csv")

    df = pd.read_csv(metrics_path)

    plot_metric = {}
    for m in metrics:
        try:
            plot_metric[m] = df[df[m].notna()][m]
        except KeyError:
            print(f"{m} not in metrics, available are {df.columns}")

    fig, ax = plt.subplots(ncols=len(metrics), figsize=(5 * len(metrics), 5))
    ax = np.atleast_1d(ax)
    for idx, (m, p) in enumerate(plot_metric.items()):
        ax[idx].plot(p)
        ax[idx].set_title(m)
    return fig


def plot_toy_regression_data(
    X_train: np.ndarray, Y_train: np.ndarray, X_gt: np.ndarray, Y_gt: np.ndarray
) -> plt.Figure:
    """Plot the toy data.

    Args:
      X_train: training inputs
      Y_train: training targets
      X_gt: X "ground truth" without noise
      Y_gt: Y "ground truth" without noise
    """
    fig, ax = plt.subplots(1)
    ax.scatter(X_gt, Y_gt, color="gray", edgecolor="black", s=5, label="test_data")
    ax.scatter(X_train, Y_train, color="blue", label="train_data")
    plt.title("Toy Regression Dataset.")
    plt.legend()
    return fig


def plot_two_moons_data(
    X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> plt.Figure:
    """Plot the two moons dataset.

    Args:
        X_train: Training data features
        Y_train: Training data labels
        X_val: Validation data features
        y_val: Validation data labels

    Returns:
        figure of twoo moons dataset
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot training data
    axs[0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap="viridis")
    axs[0].set_title("Training Set")

    # Plot validation data
    axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap="viridis")
    axs[1].set_title("Validation Set")

    return fig


def plot_predictions_classification(
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_pred: np.ndarray,
    test_grid_points,
    pred_uct: np.ndarray = None,
) -> plt.Figure:
    """Plot the classification results and the associated uncertainty.

    Args:
        X_test: The input features.
        Y_test: The true labels.
        y_pred: The predicted labels.
        test_grid_points: The grid of test points.
        pred_uct: The uncertainty of the predictions.
    """
    num_cols = 3 if pred_uct is not None else 2

    fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 6, 6))
    cm = plt.cm.plasma

    grid_size = int(np.sqrt(test_grid_points.shape[0]))
    xx = test_grid_points[:, 0].reshape(grid_size, grid_size)
    yy = test_grid_points[:, 1].reshape(grid_size, grid_size)

    # Create a scatter plot of the input features, colored by the true labels
    axs[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm, edgecolors="black")
    axs[0].set_title("True Labels")

    # Create a scatter plot of the input features, colored by the predicted labels
    axs[1].imshow(
        y_pred.reshape(grid_size, grid_size),
        alpha=0.8,
        cmap=cm,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        interpolation="bicubic",
        aspect="auto",
    )
    axs[1].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm, edgecolors="black")
    axs[1].set_title("Predicted Labels")

    if pred_uct is not None:
        # Create a scatter plot of the input features, colored by the uncertainty
        im2 = axs[2].imshow(
            pred_uct.reshape(grid_size, grid_size),
            alpha=0.8,
            cmap=cm,
            origin="lower",
            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            interpolation="bicubic",
            aspect="auto",
        )
        axs[2].scatter(
            X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm, edgecolors="black"
        )
        axs[2].set_title("Uncertainty")
        fig.colorbar(im2, ax=axs[2], fraction=0.05, pad=0.008)

    return fig


def plot_predictions_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_pred: np.ndarray,
    pred_std: np.ndarray | None = None,
    pred_quantiles: np.ndarray | None = None,
    epistemic: np.ndarray | None = None,
    aleatoric: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    title: str = None,
    show_bands: bool = False,
) -> plt.Figure:
    """Plot predictive uncertainty as well as epistemic and aleatoric separately.

    Args:
        X_train: training inputs
        Y_train: training targets
        X_test: testing inputs [batch_size, 1]
        Y_test: testing targets [batch_size, 1]
        y_pred: predicted targets [batch_size, 1]
        pred_std: predicted standard deviation
        pred_quantiles: predicted quantiles
        epistemic: epistemic uncertainy
        aleatoric: aleatoric uncertainty
        samples: samples from posterior ofh shape [batch_size, num_samples]
        title: title of plot
        show_bands: show uncertainty bands
    """
    # sort data so bands or quantiles are plotted nicely
    X_train, Y_train = sort_by(X_train, Y_train)
    X_test, Y_test, y_pred, pred_std, pred_quantiles, epistemic, aleatoric = sort_by(
        X_test, Y_test, y_pred, pred_std, pred_quantiles, epistemic, aleatoric
    )
    import textwrap

    if y_pred.ndim == 2:
        y_pred = y_pred.squeeze(-1)

    fig = plt.figure(figsize=(13, 7))
    ax0 = fig.add_subplot(1, 2, 1)
    if samples is not None:
        ax0.scatter(
            X_test, samples[:, 0], color="black", label="samples", s=0.1, alpha=0.7
        )
        for i in range(1, samples.shape[-1]):
            ax0.scatter(X_test, samples[:, i], color="black", s=0.1, alpha=0.7)

    # model predictive uncertainty bands on the left
    ax0.scatter(X_test, Y_test, color="gray", label="ground truth", s=10, alpha=0.7)
    ax0.scatter(X_train, Y_train, color="blue", label="train_data", alpha=0.5)
    ax0.scatter(X_test, y_pred, color="orange", label="predictions", alpha=0.5)

    if pred_std is not None:
        ax0.fill_between(
            X_test.squeeze(),
            y_pred - 2 * pred_std,
            y_pred + 2 * pred_std,
            alpha=0.2,
            color="tab:red",
            label=r"$2\sqrt{\mathbb{V}\,[y]}$",
        )

    if pred_quantiles is not None:
        ax0.plot(
            X_test, pred_quantiles, color="tab:red", linestyle="--", label="quantiles"
        )

    if title is not None:
        ax0.set_title(title)

    # epistemic and aleatoric uncertainty plots on right
    # epistemic uncertainty figure
    ax1 = fig.add_subplot(2, 2, 2)
    if epistemic is not None:
        if show_bands:
            ax1.scatter(
                X_test,
                Y_test,
                color="gray",
                edgecolor="black",
                label="ground truth",
                s=0.6,
            )
            ax1.fill_between(
                X_test.squeeze(),
                y_pred - epistemic,
                y_pred + epistemic,
                alpha=0.7,
                color="tab:red",
                label="Epistemic",
            )
        else:
            ax1.plot(X_test.squeeze(), epistemic, color="tab:red", label="Epistemic")

        ax1.set_title("Epistemic Uncertainty")
        ax1.legend()
    else:
        wrapped_text = textwrap.fill(
            "This Method does not quantify epistemic uncertainty.", width=30
        )
        ax1.text(
            0.5,
            0.5,
            wrapped_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    # aleatoric uncertainty figure
    ax2 = fig.add_subplot(2, 2, 4)
    if aleatoric is not None:
        if show_bands:
            ax2.scatter(
                X_test,
                Y_test,
                color="gray",
                edgecolor="black",
                label="ground truth",
                s=0.6,
            )
            ax2.fill_between(
                X_test.squeeze(),
                y_pred - aleatoric,
                y_pred + aleatoric,
                alpha=0.7,
                color="tab:red",
                label="Aleatoric",
            )
        else:
            ax2.plot(X_test.squeeze(), aleatoric, color="tab:red", label="Aleatoric")

        ax2.set_title("Aleatoric Uncertainty")
    else:
        wrapped_text = textwrap.fill(
            "This Method does not quantify aleatoric uncertainty.", width=30
        )
        ax2.text(
            0.5,
            0.5,
            wrapped_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    ax0.legend()
    return fig


def sort_by(X: np.ndarray, *args: np.ndarray) -> tuple:
    """Sort arrays based on X."""
    sort_idx = np.argsort(X.squeeze())
    return tuple(
        arg.squeeze()[sort_idx] if arg is not None else None for arg in (X, *args)
    )


def plot_calibration_uq_toolbox(
    y_pred: np.ndarray, pred_std: np.ndarray, Y_test: np.ndarray, x_test: np.ndarray
) -> plt.Figure:
    """Plot calibration from uq_toolbox.

    Taken from https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    blob/main/examples/viz_readme_figures.py.

    Args:
      y_pred: model mean predictions
      pred_std: predicted standard deviations
      Y_test: test data targets
      x_test: test data inputs
    """
    y_pred = y_pred.squeeze()
    pred_std = pred_std.squeeze()
    Y_test = Y_test.squeeze()
    x_test = x_test.squeeze()
    n_subset = min(50, y_pred.shape[0])
    mace = uct.mean_absolute_calibration_error(y_pred, pred_std, Y_test)
    ma = uct.miscalibration_area(y_pred, pred_std, Y_test)
    rmsce = uct.root_mean_squared_calibration_error(y_pred, pred_std, Y_test)

    fig, axs = plt.subplots(1, 3)

    # Make xy plot
    axs[0] = uct.plot_xy(y_pred, pred_std, Y_test, x_test, ax=axs[0])

    # Make ordered intervals plot
    axs[1] = uct.plot_intervals_ordered(
        y_pred, pred_std, Y_test, n_subset=n_subset, ax=axs[1]
    )

    # Make calibration plot
    axs[2] = uct.plot_calibration(y_pred, pred_std, Y_test, ax=axs[2])

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)

    axs[2].set_title(f"MACE: {mace:.4f}, RMSCE: {rmsce:.4f}, MA: {ma:.4f}")

    return fig
