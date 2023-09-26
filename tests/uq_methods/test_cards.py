"""Unit tests for CARDS model."""

import os
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import CARDModel




class TestCARDS:
    @pytest.fixture
    def card_linear_model(self, tmp_path: Path) -> CARDModel:
        """Instantiat a card model for 1D regression."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "card_linear.yaml"))
        # conf.uq_method["save_dir"] = str(tmp_path)
        return instantiate(conf.uq_method)

    def test_predict_step(self, card_linear_model: CARDModel) -> None:
        x_test = torch.randn(32, 1)
        out = card_linear_model.predict_step(x_test)

    def test_trainer(self, card_linear_model: CARDModel) -> None:
        """Test CARD Model with a Lightning Trainer."""
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=2,
            accelerator="cpu",
            logger=None
            # default_root_dir=card_linear_model.hparams.save_dir,
        )
        trainer.fit(model=card_linear_model, datamodule=datamodule)
        trainer.test(model=card_linear_model, datamodule=datamodule)
    



