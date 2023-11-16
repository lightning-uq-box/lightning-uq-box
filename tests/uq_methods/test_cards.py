"""Unit tests for CARDS model."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageRegressionDatamodule,
)
from lightning_uq_box.uq_methods import CARDRegression


class TestCARDS:
    @pytest.fixture
    def card_linear_model(self, tmp_path: Path) -> CARDRegression:
        """Instantiat a card model for 1D regression."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "card", "card_linear.yaml")
        )
        # conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_predict_step(self, card_linear_model: CARDRegression) -> None:
        x_test = torch.randn(32, 1)
        out = card_linear_model.predict_step(x_test)

    # def test_trainer(self, card_linear_model: CARDModel) -> None:
    #     """Test CARD Model with a Lightning Trainer."""
    #     datamodule = ToyHeteroscedasticDatamodule()
    #     trainer = Trainer(
    #         log_every_n_steps=1,
    #         max_epochs=2,
    #         accelerator="cpu",
    #         logger=None
    #         # default_root_dir=card_linear_model.hparams.save_dir,
    #     )
    #     trainer.fit(model=card_linear_model, datamodule=datamodule)
    #     trainer.test(model=card_linear_model, datamodule=datamodule)

    # image test
    @pytest.fixture
    def card_conv_model(self, tmp_path: Path) -> CARDRegression:
        """Instantiate a card model for Image regression tasks."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "card", "card_conv.yaml")
        )
        # conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_predict_step_img(self, card_conv_model: CARDRegression) -> None:
        """Test predict step with an image."""
        x_test = torch.randn(2, 3, 64, 64)
        out = card_conv_model.predict_step(x_test)

    # def test_trainer_img(self, card_conv_model: CARDModel) -> None:
    #     """Test Image Task with lightning Trainer."""
    #     datamodule = ToyImageRegressionDatamodule()
    #     trainer = Trainer(
    #         log_every_n_steps=1,
    #         max_epochs=2,
    #         accelerator="cpu",
    #         logger=None
    #         # default_root_dir=card_linear_model.hparams.save_dir,
    #     )
    #     trainer.fit(model=card_conv_model, datamodule=datamodule)
    #     trainer.test(model=card_conv_model, datamodule=datamodule)
