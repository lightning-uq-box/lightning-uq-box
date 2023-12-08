# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Utilities for visualizing UQ-Predictions."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from lightning_uq_box.eval_utils import compute_quantiles_from_std
from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.viz_utils.visualization_tools import (
    plot_calibration_uq_toolbox,
    plot_predictions_classification,
    plot_predictions_regression,
    plot_toy_regression_data,
    plot_training_metrics,
    plot_two_moons_data,
)


class TestRegressionVisualization:
    @pytest.fixture(
        params=[
            (
                "tests/configs/regression/mc_dropout_nll.yaml",
                "tests/configs/regression/toy_regression.yaml",
            )
        ]
    )
    def exp_run(self, request, tmp_path: Path):
        model_config_path, data_config_path = request.param
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "2",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.logger",
            "CSVLogger",
            "--trainer.logger.save_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

        # assert predictions are saved
        assert os.path.exists(
            os.path.join(cli.trainer.default_root_dir, cli.model.pred_file_name)
        )
        return cli

    def test_plot_training_metrics(self, exp_run):
        """Test plot_training_metrics."""
        fig = plot_training_metrics(
            os.path.join(exp_run.trainer.default_root_dir, "lightning_logs"), "RMSE"
        )
        assert fig is not None
        plt.close()

    def test_plot_toy_regression_data(self, exp_run):
        """Test plot_toy_regression_data."""
        fig = plot_toy_regression_data(
            exp_run.datamodule.X_train,
            exp_run.datamodule.y_train,
            exp_run.datamodule.X_test,
            exp_run.datamodule.y_test,
        )
        assert fig is not None
        plt.close()

    @pytest.mark.parametrize("show_bands", [True, False])
    def test_plot_predictions_regression(self, exp_run, show_bands):
        """Test plot_predictions_regression."""
        pred_dict = exp_run.model.predict_step(exp_run.datamodule.X_test)

        # Plot predictions
        fig = plot_predictions_regression(
            X_train=exp_run.datamodule.X_train,
            y_train=exp_run.datamodule.y_train,
            X_test=exp_run.datamodule.X_test,
            y_test=exp_run.datamodule.y_test,
            y_pred=pred_dict["pred"],
            title="Test Plot",
            show_bands=show_bands,
        )
        assert fig is not None
        plt.close()

    def test_plot_predictions_regression_epistemic_and_aleatoric(self, exp_run):
        """Test plot_predictions_regression."""
        pred_dict = exp_run.model.predict_step(exp_run.datamodule.X_test)

        # Plot predictions
        fig = plot_predictions_regression(
            X_train=exp_run.datamodule.X_train,
            y_train=exp_run.datamodule.y_train,
            X_test=exp_run.datamodule.X_test,
            y_test=exp_run.datamodule.y_test,
            y_pred=pred_dict["pred"],
            pred_std=pred_dict["pred_uct"],
            epistemic=pred_dict["epistemic_uct"],
            aleatoric=pred_dict["aleatoric_uct"],
            title="Test Plot",
            show_bands=True,
        )
        assert fig is not None
        plt.close()

    def test_plot_quantiles(self, exp_run):
        """Test plot_quantiles."""
        pred_dict = exp_run.model.predict_step(exp_run.datamodule.X_test)

        quantiles = compute_quantiles_from_std(
            pred_dict["pred"].numpy(),
            pred_dict["pred_uct"].numpy(),
            quantiles=[0.1, 0.5, 0.9],
        )
        # Plot predictions
        fig = plot_predictions_regression(
            X_train=exp_run.datamodule.X_train,
            y_train=exp_run.datamodule.y_train,
            X_test=exp_run.datamodule.X_test,
            y_test=exp_run.datamodule.y_test,
            y_pred=pred_dict["pred"],
            pred_quantiles=quantiles,
            title="Test Plot",
            show_bands=True,
        )
        assert fig is not None
        plt.close()

    def test_plot_calibration(self, exp_run):
        """Test plot calibration."""

        pred_dict = exp_run.model.predict_step(exp_run.datamodule.X_test)

        fig = plot_calibration_uq_toolbox(
            y_pred=pred_dict["pred"].numpy(),
            pred_std=pred_dict["pred_uct"].numpy(),
            y_test=exp_run.datamodule.y_test.numpy(),
            x_test=exp_run.datamodule.X_test.numpy(),
        )
        assert fig is not None
        plt.close()


class TestClassificationVisualization:
    @pytest.fixture(
        params=[
            (
                "tests/configs/classification/mc_dropout.yaml",
                "tests/configs/classification/toy_classification.yaml",
            )
        ]
    )
    def exp_run(self, request, tmp_path: Path):
        model_config_path, data_config_path = request.param
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.accelerator",
            "cpu",
            "--trainer.max_epochs",
            "2",
            "--trainer.log_every_n_steps",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.logger",
            "CSVLogger",
            "--trainer.logger.save_dir",
            str(tmp_path),
        ]

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

        # TODO: assert predictions are saved
        # assert os.path.exists(
        #     os.path.join(cli.trainer.default_root_dir, cli.model.pred_file_name)
        # )
        return cli

    def test_plot_two_moons_data(self, exp_run):
        """Test plot_two_moons_data."""
        fig = plot_two_moons_data(
            exp_run.datamodule.X_train,
            exp_run.datamodule.y_train,
            exp_run.datamodule.X_test,
            exp_run.datamodule.y_test,
        )
        assert fig is not None
        plt.close()

    def test_plot_predictions_classification(self, exp_run):
        """Test plot_predictions_classification."""
        test_grid_points = exp_run.datamodule.test_grid_points
        pred_dict = exp_run.model.predict_step(test_grid_points)

        # Plot predictions
        fig = plot_predictions_classification(
            X_test=exp_run.datamodule.X_test.numpy(),
            y_test=exp_run.datamodule.y_test.numpy(),
            y_pred=pred_dict["pred"].argmax(-1).numpy(),
            test_grid_points=test_grid_points.numpy(),
        )
        assert fig is not None
        plt.close()

    def test_plot_predictions_classification_with_uq(self, exp_run):
        """Test plot_predictions_classification."""
        test_grid_points = exp_run.datamodule.test_grid_points
        pred_dict = exp_run.model.predict_step(test_grid_points)

        # Plot predictions
        fig = plot_predictions_classification(
            X_test=exp_run.datamodule.X_test.numpy(),
            y_test=exp_run.datamodule.y_test.numpy(),
            y_pred=pred_dict["pred"].argmax(-1).numpy(),
            test_grid_points=test_grid_points.numpy(),
            pred_uct=pred_dict["pred_uct"].numpy(),
        )
        assert fig is not None
        plt.close()
