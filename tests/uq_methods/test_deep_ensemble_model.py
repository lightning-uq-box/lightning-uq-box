"""Unit tests Deep Ensemble Model."""

import os
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import DeepEnsembleModel


class TestBaseEnsembleModel:
    @pytest.fixture(params=["deep_ensemble_nll.yaml", "deep_ensemble_mse.yaml"])
    def ensemble_model(self, tmp_path: Path, request: SubRequest) -> DeepEnsembleModel:
        """Create a Deep Ensemble Model being used for tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", request.param))

        model_paths = []
        n_ensemble_members = conf.post_processing["n_ensemble_members"]
        for i in range(n_ensemble_members):
            # set temporary save directory for logging
            save_path = os.path.join(tmp_path, f"ensemble_{i}")
            conf.uq_method["save_dir"] = save_path

            # instantiate datamodule
            datamodule = ToyHeteroscedasticDatamodule()

            model = instantiate(conf.uq_method)

            # Instantiate trainer
            trainer = Trainer(
                log_every_n_steps=1, max_epochs=2, default_root_dir=save_path
            )
            trainer.fit(model=model, datamodule=datamodule)

            # save path to load them for ensemble
            model_paths.append(
                os.path.join(
                    save_path,
                    "lightning_logs",
                    "version_0",
                    "checkpoints",
                    os.listdir(
                        os.path.join(
                            trainer.default_root_dir,
                            "lightning_logs",
                            "version_0",
                            "checkpoints",
                        )
                    )[0],
                )
            )

        ensemble_members = []
        for ckpt_path in model_paths:
            ensemble_members.append({"base_model": model, "ckpt_path": ckpt_path})

        return DeepEnsembleModel(
            n_ensemble_members,
            ensemble_members,
            save_dir=os.path.join(tmp_path, "prediction"),
        )

    def test_ensemble_forward(self, ensemble_model: DeepEnsembleModel) -> None:
        """Test ensemble predict step."""
        X = torch.randn(5, 1)
        out = ensemble_model.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_ensemble_trainer_test(self, ensemble_model: DeepEnsembleModel) -> None:
        """Test ensemble test step with lightning Trainer."""
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=ensemble_model.hparams.save_dir,
        )
        trainer.test(ensemble_model, datamodule.test_dataloader())
