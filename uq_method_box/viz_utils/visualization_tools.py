"""Visualization utils for Regression Uncertainty."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import uncertainty_toolbox as uct


def plot_toy_data(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    """Plot the toy data.

    Args:
      X_train: training inputs
      y_train: training targets
      X_test: testing inputs
      y_test: testing targets
    """
    fig, ax = plt.subplots(1)
    ax.scatter(X_test, y_test, color="gray", edgecolor="black", s=5, label="test_data")
    ax.scatter(X_train, y_train, color="blue", label="train_data")
    plt.legend()
    plt.show()


def plot_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    pred_std: Optional[np.ndarray] = None,
    pred_quantiles: Optional[np.ndarray] = None,
    epistemic: Optional[np.ndarray] = None,
    aleatoric: Optional[np.ndarray] = None,
    title: str = None,
) -> None:
    """Plot predictive uncertainty as well as epistemic and aleatoric separately.

    Args:
      X_train: training inputs
      y_train: training targets
      X_test: testing inputs
      y_test: testing targets
      y_pred: predicted targets
      pred_std: predicted standard deviation
      pred_quantiles: predicted quantiles
      epistemic: epistemic uncertainy
      aleatoric: aleatoric uncertainty
    """
    # fig, ax = plt.subplots(ncols=2)
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)

    # model predictive uncertainty bands on the left
    ax0.scatter(X_test, y_test, color="gray", label="ground truth", s=0.5)
    ax0.scatter(X_train, y_train, color="blue", label="train_data")
    ax0.scatter(X_test, y_pred, color="orange", label="predictions")

    if pred_std is not None:
        ax0.fill_between(
            X_test.squeeze(),
            y_pred - pred_std,
            y_pred + pred_std,
            alpha=0.3,
            color="tab:red",
            label=r"$\sqrt{\mathbb{V}\,[y]}$",
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
        ax1.scatter(
            X_test, y_test, color="gray", edgecolor="black", label="ground truth", s=0.6
        )
        ax1.set_title("Epistemic Uncertainty")
        ax1.fill_between(
            X_test.squeeze(),
            y_pred - epistemic,
            y_pred + epistemic,
            alpha=0.3,
            color="tab:red",
            label="Epistemic",
        )
        ax1.set_title("Epistemic Uncertainty")
        ax1.legend()
    else:
        ax1.text(
            0.5,
            0.5,
            "This Method does not quantify epistemic uncertainty.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=15,
        )

    # aleatoric uncertainty figure
    ax2 = fig.add_subplot(2, 2, 4)
    if aleatoric is not None:
        ax2.scatter(
            X_test, y_test, color="gray", edgecolor="black", label="ground truth", s=0.6
        )
        ax2.fill_between(
            X_test.squeeze(),
            y_pred - aleatoric,
            y_pred + aleatoric,
            alpha=0.3,
            color="tab:red",
            label="Aleatoric",
        )
        ax2.set_title("Aleatoric Uncertainty")
    else:
        ax2.text(
            0.5,
            0.5,
            "This Method does not quantify aleatoric uncertainty.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=15,
        )

    ax0.legend()
    # plt.show()
    return fig


def plot_calibration_uq_toolbox(
    y_pred: np.ndarray, pred_std: np.ndarray, y_test: np.ndarray, x_test: np.ndarray
) -> None:
    """Plot calibration from uq_toolbox.

    Taken from https://github.com/uncertainty-toolbox/uncertainty-toolbox/
    blob/main/examples/viz_readme_figures.py.

    Args:
      y_pred: model mean predictions
      pred_std: predicted standard deviations
      y_test: test data targets
      x_test: test data inputs
    """
    fig = plt.figure()
    y_pred = y_pred.squeeze()
    pred_std = pred_std.squeeze()
    y_test = y_test.squeeze()
    x_test = x_test.squeeze()
    n_subset = 50
    mace = uct.mean_absolute_calibration_error(y_pred, pred_std, y_test)
    ma = uct.miscalibration_area(y_pred, pred_std, y_test)
    rmsce = uct.root_mean_squared_calibration_error(y_pred, pred_std, y_test)

    fig, axs = plt.subplots(1, 3)

    # Make xy plot
    axs[0] = uct.plot_xy(y_pred, pred_std, y_test, x_test, ax=axs[0])

    # Make ordered intervals plot
    axs[1] = uct.plot_intervals_ordered(
        y_pred, pred_std, y_test, n_subset=n_subset, ax=axs[1]
    )

    # Make calibration plot
    axs[2] = uct.plot_calibration(y_pred, pred_std, y_test, ax=axs[2])

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)

    axs[2].set_title(f"MACE: {mace:.4f}, RMSCE: {rmsce:.4f}, MA: {ma:.4f}")

    plt.show()
