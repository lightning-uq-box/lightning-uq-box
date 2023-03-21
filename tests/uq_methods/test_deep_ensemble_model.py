"""Unit tests Deep Ensemble Model."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BaseModel, DeepEnsembleModel


class TestBaseEnsembleModel:
    @pytest.fixture
    def ensemble_model(self, tmp_path: Path) -> DeepEnsembleModel:
        """Create a Deep Ensemble Model being used for tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "deep_ensemble.yaml"))
        conf_dict = OmegaConf.to_object(conf)

        model_paths = []
        for i in range(conf_dict["model"]["ensemble_members"]):
            # set temporary save directory for logging
            conf_dict["experiment"] = {}
            save_path = os.path.join(tmp_path, f"ensemble_{i}")
            conf_dict["experiment"]["save_dir"] = save_path

            # instantiate datamodule
            datamodule = ToyHeteroscedasticDatamodule()

            model = BaseModel(
                MLP,
                model_args=conf_dict["model"]["model_args"],
                lr=1e-3,
                loss_fn="mse",
                save_dir=save_path,
            )

            # Instantiate trainer
            trainer = Trainer(
                log_every_n_steps=1, max_epochs=1, default_root_dir=save_path
            )
            trainer.fit(model=model, datamodule=datamodule)

            # save path to load them for ensemble
            model_paths.append(
                os.path.join(
                    save_path,
                    "lightning_logs",
                    "version_0",
                    "checkpoints",
                    "epoch=0-step=1.ckpt",
                )
            )

        ensemble_members = []
        for ckpt_path in model_paths:
            ensemble_members.append({"model_class": BaseModel, "ckpt_path": ckpt_path})

        return DeepEnsembleModel(
            ensemble_members, save_dir=os.path.join(tmp_path, "prediction")
        )

    def test_ensemble_forward(self, ensemble_model: DeepEnsembleModel) -> None:
        """Test ensemble predict step."""
        print(ensemble_model.hparams.ensemble_members)
        X = torch.randn(5, 1)
        out = ensemble_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_ensemble_trainer_test(self, ensemble_model: DeepEnsembleModel) -> None:
        """Test ensemble test step with lightning Trainer."""
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=ensemble_model.hparams.save_dir,
        )
        trainer.test(ensemble_model, datamodule.test_dataloader())
