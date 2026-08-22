# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Laplace Tuning procedure."""

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import minimal_trainer_kwargs
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

data_config_path = "tests/configs/image_regression/toy_image_regression.yaml"

model_config_path = "tests/configs/image_regression/laplace_nn.yaml"


@pytest.fixture
def common_setup(tmp_path: Path, accelerator_config):
    model_conf = OmegaConf.load(model_config_path)  # type: ignore[index]
    data_conf = OmegaConf.load(data_config_path)  # type: ignore[index]

    sigma_val = 1.234
    precision_val = 1.789
    model_conf["uq_method"]["laplace_model"]["sigma_noise"] = sigma_val
    model_conf["uq_method"]["laplace_model"]["prior_precision"] = precision_val

    trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))

    return trainer, model_conf, data_conf, sigma_val, precision_val


@pytest.mark.parametrize(
    "tune_sigma_noise, tune_prior_precision",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_tuning(common_setup, tune_sigma_noise, tune_prior_precision):
    trainer, model_conf, data_conf, sigma_val, precision_val = common_setup
    model_conf["uq_method"]["tune_sigma_noise"] = tune_sigma_noise
    model_conf["uq_method"]["tune_prior_precision"] = tune_prior_precision

    model = instantiate(model_conf.uq_method)
    datamodule = instantiate(data_conf.data)

    trainer.test(model, datamodule)
    assert np.isclose(model.laplace_model.sigma_noise.item(), sigma_val, atol=1e-8) == (
        not tune_sigma_noise
    )
    assert np.isclose(
        model.laplace_model.prior_precision.item(), precision_val, atol=1e-8
    ) == (not tune_prior_precision)

    x = torch.randn(4, 3, 32, 32).to(model.device)
    pred_dict = model.predict_step(x)
    assert "pred_uct" in pred_dict
