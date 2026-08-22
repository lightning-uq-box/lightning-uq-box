"""Shared pytest configuration and lightweight integration-test settings."""

from pathlib import Path

import pytest


def minimal_trainer_kwargs(
    accelerator_config: dict,
    default_root_dir: str | Path,
    *,
    max_epochs: int = 1,
    checkpoints: bool = False,
    **overrides,
) -> dict:
    """Return Trainer settings for one-batch integration smoke tests.

    Checkpoints are opt-in because only tests that pass ``ckpt_path="best"`` need
    them. Keeping the profile here makes the resource bound explicit and shared.
    """
    kwargs = {
        "accelerator": accelerator_config["accelerator"],
        "devices": accelerator_config["devices"],
        "max_epochs": max_epochs,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": checkpoints,
        "logger": False,
        "log_every_n_steps": 1,
        "default_root_dir": str(default_root_dir),
    }
    kwargs.update(overrides)
    return kwargs


def minimal_cli_overrides(
    accelerator_config: dict,
    default_root_dir: str | Path,
    *,
    max_epochs: int = 1,
    checkpoints: bool = False,
    logging: bool = False,
) -> list[str]:
    """Return LightningCLI overrides matching :func:`minimal_trainer_kwargs`."""
    overrides = [
        "--trainer.accelerator",
        accelerator_config["accelerator"],
        "--trainer.devices",
        str(accelerator_config["devices"]),
        "--trainer.max_epochs",
        str(max_epochs),
        "--trainer.limit_train_batches",
        "1",
        "--trainer.limit_val_batches",
        "1",
        "--trainer.limit_test_batches",
        "1",
        "--trainer.num_sanity_val_steps",
        "0",
        "--trainer.enable_progress_bar",
        "False",
        "--trainer.enable_model_summary",
        "False",
        "--trainer.enable_checkpointing",
        str(checkpoints),
        "--trainer.log_every_n_steps",
        "1",
        "--trainer.default_root_dir",
        str(default_root_dir),
    ]
    if logging:
        overrides.extend(
            [
                "--trainer.logger",
                "CSVLogger",
                "--trainer.logger.save_dir",
                str(default_root_dir),
            ]
        )
    else:
        overrides.extend(["--trainer.logger", "False"])
    return overrides


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--accelerator",
        action="store",
        default="cpu",
        help="Accelerator to use: cpu or gpu",
    )
    parser.addoption(
        "--devices",
        action="store",
        default="auto",
        help="Number of devices to use (e.g., '0')",
    )


@pytest.fixture(scope="session")
def accelerator_config(request):
    """Fixture to get accelerator configuration from command line."""
    accelerator = request.config.getoption("--accelerator")
    devices = request.config.getoption("--devices")
    if devices != "none" and devices != "auto":
        devices = [int(devices)]
    return {"accelerator": accelerator, "devices": devices}
