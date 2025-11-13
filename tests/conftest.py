"""Shared pytest configuration."""

import pytest


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
