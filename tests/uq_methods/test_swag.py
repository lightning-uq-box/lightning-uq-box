"""Test SWAG."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from uq_method_box.datamodules import ToyHeteroscedasticDatamodule
from uq_method_box.uq_methods import DeterministicGaussianModel, SWAGModel

# TODO
# Tests for unused train and validation step
# Tests with batchnorm, maybe resnet model
# Tests for partial stochasticity


class TestSWAGModel:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def base_model_tabular(self, tmp_path: Path) -> DeterministicGaussianModel:
        """Create a Base model being used for different tests."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "swag.yaml"))

        # train the model with a trainer
        model = instantiate(conf.uq_method, save_dir=tmp_path)
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1, max_epochs=1, default_root_dir=model.hparams.save_dir
        )
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture(params=[[-1], ["model.0"]])
    def swag_model_tabular(
        self,
        base_model_tabular: DeterministicGaussianModel,
        tmp_path: Path,
        request: SubRequest,
    ) -> SWAGModel:
        """Create SWAG model from an underlying model."""
        conf = OmegaConf.load(os.path.join("tests", "configs", "swag.yaml"))
        conf.post_processing["save_dir"] = str(tmp_path)
        datamodule = ToyHeteroscedasticDatamodule()
        train_loader = datamodule.train_dataloader()
        return instantiate(
            conf.post_processing,
            model=base_model_tabular.model,
            train_loader=train_loader,
            part_stoch_module_names=request.param,
        )

    def test_tabular_predict_step(self, swag_model_tabular: SWAGModel) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = swag_model_tabular.num_inputs
        X = torch.randn(5, n_inputs)
        out = swag_model_tabular.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["mean"], np.ndarray)
        assert out["mean"].shape[0] == 5

    def test_tabular_trainer(self, swag_model_tabular: SWAGModel) -> None:
        """Test SWAG Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(
            log_every_n_steps=1,
            max_epochs=1,
            default_root_dir=swag_model_tabular.hparams.save_dir,
        )
        trainer.test(model=swag_model_tabular, datamodule=datamodule)
