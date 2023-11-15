"""Test SWAG."""

import os
from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    ToyImageClassificationDatamodule,
    ToyImageRegressionDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.uq_methods import (
    DeterministicClassification,
    MVERegression,
    SWAGClassification,
    SWAGRegression,
)

# TODO
# Tests with batchnorm
# need to test that we are able to conformalize all models?


class TestSWAGRegression:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def base_model_tabular(self, tmp_path: Path) -> MVERegression:
        """Create a Base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_regression.yaml")
        )
        # train the model with a trainer
        model = instantiate(conf.uq_method)
        datamodule = ToyHeteroscedasticDatamodule(n_train=64, n_true=64, batch_size=64)
        trainer = Trainer(log_every_n_steps=1, max_epochs=1, default_root_dir=tmp_path)
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture(params=[None, [-1], ["model.0"]])
    def swag_model_tabular(
        self, base_model_tabular: MVERegression, tmp_path: Path, request: SubRequest
    ) -> SWAGRegression:
        """Create SWAG model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_regression.yaml")
        )
        conf.post_processing["part_stoch_module_names"] = request.param

        datamodule = ToyHeteroscedasticDatamodule()

        swag_model = instantiate(conf.post_processing, model=base_model_tabular.model)

        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model, datamodule=datamodule)

        return swag_model

    def test_tabular_predict_step(self, swag_model_tabular: SWAGRegression) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = swag_model_tabular.num_input_dims
        X = torch.randn(5, n_inputs)
        out = swag_model_tabular.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_tabular_trainer(self, swag_model_tabular: SWAGRegression) -> None:
        """Test SWAG Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyHeteroscedasticDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.test(model=swag_model_tabular, datamodule=datamodule)

    @pytest.fixture
    def base_model_image(self) -> MVERegression:
        """Create a Base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_regression.yaml")
        )
        # train the model with a trainer
        model = instantiate(
            conf.uq_method,
            model=timm.create_model("resnet18", in_chans=3, num_classes=2),
        )
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model, datamodule)

        return model

    # test for image task
    @pytest.fixture(params=[None, [-1], ["layer4.1.conv1", "layer4.1.conv2"]])
    def swag_model_image(
        self, base_model_image: MVERegression, request: SubRequest
    ) -> SWAGRegression:
        """Create SWAG model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_regression.yaml")
        )
        conf.post_processing["part_stoch_module_names"] = request.param
        datamodule = ToyImageRegressionDatamodule()

        swag_model = instantiate(conf.post_processing, model=base_model_image.model)

        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model, datamodule=datamodule)

        return swag_model

    # tests for image data
    def test_forward_image(self, swag_model_image: SWAGRegression) -> None:
        """Test forward pass of model."""
        X = torch.randn(2, 3, 32, 32)
        out = swag_model_image(X)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 2
        assert out.shape[-1] == 2

    def test_predict_step_image(self, swag_model_image: SWAGRegression) -> None:
        """Test predict step outside of Lightning Trainer."""
        X = torch.randn(2, 3, 32, 32)
        out = swag_model_image.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 2

    def test_trainer_image(self, swag_model_image: SWAGRegression) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model_image, datamodule=datamodule)


# TESTS FOR CLASSIFICATION
class TestSWAGClassification:
    # TODO need to test that we are able to conformalize all models
    @pytest.fixture
    def base_model_tabular(
        self, tmp_path: Path, request=SubRequest
    ) -> DeterministicClassification:
        """Create a Base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_classification.yaml")
        )
        # train the model with a trainer
        model = instantiate(conf.uq_method)
        datamodule = TwoMoonsDataModule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1, default_root_dir=tmp_path)
        trainer.fit(model, datamodule)

        return model

    @pytest.fixture(params=[None, [-1], ["model.0"]])
    def swag_model_tabular(
        self, base_model_tabular: DeterministicClassification, request: SubRequest
    ) -> SWAGClassification:
        """Create SWAG model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_classification.yaml")
        )
        conf.post_processing["part_stoch_module_names"] = request.param

        datamodule = TwoMoonsDataModule()

        swag_model = instantiate(conf.post_processing, model=base_model_tabular.model)

        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model, datamodule=datamodule)

        return swag_model

    def test_tabular_predict_step(self, swag_model_tabular: SWAGClassification) -> None:
        """Test predict step outside of Lightning Trainer."""
        n_inputs = swag_model_tabular.num_input_dims
        X = torch.randn(5, n_inputs)
        out = swag_model_tabular.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 5

    def test_tabular_trainer(self, swag_model_tabular: SWAGClassification) -> None:
        """Test SWAG Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = TwoMoonsDataModule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.test(model=swag_model_tabular, datamodule=datamodule)

    @pytest.fixture
    def base_model_image(self, request=SubRequest) -> MVERegression:
        """Create a Base model being used for different tests."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_classification.yaml")
        )
        # train the model with a trainer
        model = instantiate(
            conf.uq_method,
            model=timm.create_model("resnet18", in_chans=3, num_classes=2),
        )
        datamodule = ToyImageClassificationDatamodule()
        trainer = Trainer(log_every_n_steps=1, max_epochs=1)
        trainer.fit(model, datamodule)

        return model

    # test for image task
    @pytest.fixture(params=[None, [-1], ["layer4.1.conv1", "layer4.1.conv2"]])
    def swag_model_image(
        self, base_model_image: DeterministicClassification, request: SubRequest
    ) -> SWAGClassification:
        """Create SWAG model from an underlying model."""
        conf = OmegaConf.load(
            os.path.join("tests", "configs", "swag", "swag_classification.yaml")
        )
        conf.post_processing["part_stoch_module_names"] = request.param
        datamodule = ToyImageClassificationDatamodule()

        swag_model = instantiate(conf.post_processing, model=base_model_image.model)

        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model, datamodule=datamodule)

        return swag_model

    # tests for image data
    def test_forward_image(self, swag_model_image: SWAGClassification) -> None:
        """Test forward pass of model."""
        X = torch.randn(2, 3, 32, 32)
        out = swag_model_image(X)
        assert isinstance(out, Tensor)
        assert out.shape[0] == 2
        assert out.shape[-1] == 2

    def test_predict_step_image(self, swag_model_image: SWAGClassification) -> None:
        """Test predict step outside of Lightning Trainer."""
        X = torch.randn(2, 3, 32, 32)
        out = swag_model_image.predict_step(X)
        assert isinstance(out, dict)
        assert isinstance(out["pred"], Tensor)
        assert out["pred"].shape[0] == 2

    def test_trainer_image(self, swag_model_image: SWAGClassification) -> None:
        """Test Model with a Lightning Trainer."""
        # instantiate datamodule
        datamodule = ToyImageRegressionDatamodule()
        trainer = Trainer(logger=False, max_epochs=1)
        trainer.test(model=swag_model_image, datamodule=datamodule)
