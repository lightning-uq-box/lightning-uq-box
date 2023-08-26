"""Unit tests Multi-SWAG Model."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import MultiSWAG


class TestMultiSWAG:
    @pytest.fixture
    def multi_swag_model(self, tmp_path: Path) -> MultiSWAG:
        """Create a Deep Ensemble Model being used for tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "multi_swag.yaml"))

        model_paths = []
        n_ensemble_members = 3
        for i in range(n_ensemble_members):
            # set temporary save directory for logging
            save_path = os.path.join(tmp_path, f"ensemble_{i}")
            conf.uq_method["save_dir"] = save_path

            # instantiate datamodule
            datamodule = ToyHeteroscedasticDatamodule()

            det_model = instantiate(conf.uq_method)

            # Instantiate trainer and fit deterministic model
            trainer = Trainer(
                log_every_n_steps=1, max_epochs=2, default_root_dir=save_path
            )
            trainer.fit(model=det_model, datamodule=datamodule)

            # fit SWAG
            swag_model = instantiate(
                conf.post_processing,
                model=det_model.model,
                train_loader=datamodule.train_dataloader(),
                save_dir=save_path,
            )
            trainer = Trainer(
                log_every_n_steps=1, max_epochs=2, default_root_dir=save_path
            )
            trainer.test(model=swag_model, datamodule=datamodule)
            save_ckpt_path = os.path.join(save_path, f"swag_{i}.ckpt")
            trainer.save_checkpoint(save_ckpt_path)

            # save path to SWAG model load them for ensemble
            model_paths.append(save_ckpt_path)

        ensemble_members = []
        for ckpt_path in model_paths:
            ensemble_members.append({"base_model": swag_model, "ckpt_path": ckpt_path})

        return MultiSWAG(
            n_ensemble_members,
            ensemble_members,
            save_dir=os.path.join(tmp_path, "prediction"),
        )

    def test_ensemble_forward(self, multi_swag_model: MultiSWAG) -> None:
        """Test ensemble predict step."""
        X = torch.randn(5, 1)
        out = multi_swag_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_ensemble_trainer_test(self, multi_swag_model: MultiSWAG) -> None:
        """Test ensemble test step with lightning Trainer."""
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=multi_swag_model.hparams.save_dir,
        )
        trainer.test(multi_swag_model, datamodule.test_dataloader())
