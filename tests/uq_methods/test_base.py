"""Unit tests for base class."""

import os
import sys
import tempfile

import pytest
import torch
from pytorch_lightning import Trainer

# required to make the path visible to import the tools
# this will change in public notebooks to be "pip install uq-regression-box"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.models import MLP
from uq_method_box.uq_methods import BaseModel, EnsembleModel


class TestBaseModel:
    @pytest.mark.parametrize("name", ["base"])
    def test_trainer(self, name: str) -> None:
        conf = OmegaConf.load(os.path.join("tests", "configs", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf)

        # set temporary save directory for logging
        conf_dict["experiment"] = {}
        conf_dict["experiment"]["save_dir"] = tempfile.gettempdir()

        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()

        model = BaseModel(MLP(**conf_dict["model"]["mlp"]), conf_dict)

        # Instantiate trainer
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)

        trainer.test(model=model, datamodule=datamodule)


class TestBaseEnsembleModel:
    @pytest.mark.parametrize("name", ["base_ensemble"])
    def test_trainer(self, name: str) -> None:
        conf = OmegaConf.load(os.path.join("tests", "configs", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf)

        exp_root = tempfile.gettempdir()
        model_paths = []
        for i in range(conf_dict["model"]["ensemble_members"]):
            # set temporary save directory for logging
            conf_dict["experiment"] = {}
            save_path = os.path.join(exp_root, f"ensemble_{i}")
            conf_dict["experiment"]["save_dir"] = save_path

            # instantiate datamodule
            datamodule = ToyHeteroscedasticDatamodule()

            mlp = MLP(**conf_dict["model"]["mlp"])
            model = BaseModel(mlp, conf_dict)

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

        # wrap Deep Ensemble Model
        ensemble_members = []
        for ckpt_path in model_paths:
            import pdb

            pdb.set_trace()
            checkpoint = torch.load(ckpt_path)
            ensemble_members.append(BaseModel(mlp).load_from_checkpoint(ckpt_path))

        ensemble_dict = OmegaConf.to_object(conf)
        ensemble_dict["experiment"]["save_dir"] = os.path.join(exp_root, "prediction")
        model = EnsembleModel(ensemble_dict, ensemble_members)

        trainer.test(model=model, datamodule=datamodule)
