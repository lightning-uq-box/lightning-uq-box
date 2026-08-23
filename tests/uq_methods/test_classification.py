# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test Classification Tasks."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from conftest import minimal_cli_overrides, minimal_trainer_kwargs
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytest import TempPathFactory
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.main import get_uq_box_cli
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (
    DeepEnsembleClassification,
    DeterministicClassification,
)

model_config_paths = [
    "tests/configs/classification/mc_dropout.yaml",
    "tests/configs/classification/bnn_vi_elbo.yaml",
    "tests/configs/classification/swag.yaml",
    "tests/configs/classification/sgld.yaml",
    "tests/configs/classification/dkl.yaml",
    "tests/configs/classification/due.yaml",
    "tests/configs/classification/card.yaml",
    "tests/configs/classification/sngp.yaml",
    "tests/configs/classification/vbll.yaml",
    "tests/configs/classification/masked_ensemble.yaml",
    "tests/configs/classification/zigzag.yaml",
]

data_config_paths = ["tests/configs/classification/toy_classification.yaml"]


class TestClassificationTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
        ] + minimal_cli_overrides(
            accelerator_config,
            tmp_path,
            max_epochs=2
            if "swag" in model_config_path or "sgld" in model_config_path
            else 1,
            checkpoints=True,
        )

        cli = get_uq_box_cli(args)
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

        # assert predictions are saved
        assert os.path.exists(
            os.path.join(cli.trainer.default_root_dir, cli.model.pred_file_name)
        )


posthoc_config_paths = [
    "tests/configs/classification/temp_scaling.yaml",
    "tests/configs/classification/raps.yaml",
    "tests/configs/classification/raps_mc_dropout.yaml",
]


class TestPosthoc:
    @pytest.mark.parametrize("model_config_path", posthoc_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        args = [
            "--config",
            model_config_path,
            "--config",
            data_config_path,
            "--trainer.inference_mode",
            "False",
        ] + minimal_cli_overrides(accelerator_config, tmp_path)

        cli = get_uq_box_cli(args)
        model = cli.model
        # use validation for testing, should be calibration loader for conformal
        cli.trainer.fit(model, train_dataloaders=cli.datamodule.val_dataloader())
        cli.trainer.test(model, datamodule=cli.datamodule)


ensemble_model_config_paths = ["tests/configs/classification/mc_dropout.yaml"]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(
        self, request, tmp_path_factory: TempPathFactory, accelerator_config: dict
    ) -> list[dict[str, Any]]:
        model_config_path, data_config_path = request.param
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(2):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            args = [
                "--config",
                model_config_path,
                "--config",
                data_config_path,
            ] + minimal_cli_overrides(accelerator_config, tmp_path, checkpoints=True)

            cli = get_uq_box_cli(args)
            cli.trainer.fit(cli.model, cli.datamodule)

            ckpt_cb = cli.trainer.checkpoint_callback
            assert isinstance(ckpt_cb, ModelCheckpoint)
            ckpt_file = ckpt_cb.best_model_path
            assert ckpt_file
            ckpt_paths.append({"base_model": cli.model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self,
        ensemble_members_dict: list[dict[str, Any]],
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsembleClassification(ensemble_members_dict, 2)

        datamodule = TwoMoonsDataModule(batch_size=4, n_samples=20)

        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))

        trainer.test(ensemble_model, datamodule=datamodule)


frozen_config_paths = [
    "tests/configs/classification/mc_dropout.yaml",
    "tests/configs/classification/bnn_vi_elbo.yaml",
    "tests/configs/classification/dkl.yaml",
    "tests/configs/classification/due.yaml",
    "tests/configs/classification/sngp.yaml",
]


class TestFrozenBackbone:
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(self, model_config_path: str) -> None:
        cli = get_uq_box_cli(
            ["--config", model_config_path, "--model.freeze_backbone", "True"]
        )
        model = cli.model
        try:
            assert not all(
                [param.requires_grad for param in model.model.model[0].parameters()]
            )
            assert all(
                [param.requires_grad for param in model.model.model[-1].parameters()]
            )
        except AttributeError:
            # check that entire feature extractor is frozen
            assert not all(
                [param.requires_grad for param in model.feature_extractor.parameters()]
            )


class ToyMultilabelDataset(Dataset):
    """Toy multilabel dataset with multi-hot targets."""

    def __init__(self, n_samples: int = 16, n_labels: int = 3) -> None:
        generator = torch.Generator().manual_seed(0)
        self.inputs = torch.randn(n_samples, 2, generator=generator)
        self.targets = (
            torch.rand(n_samples, n_labels, generator=generator) > 0.5
        ).float()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"input": self.inputs[index], "target": self.targets[index]}


class ToyMultilabelDataModule(LightningDataModule):
    """Toy multilabel datamodule with multi-hot targets."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset = ToyMultilabelDataset()

    def _dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=4)

    train_dataloader = _dataloader
    val_dataloader = _dataloader
    test_dataloader = _dataloader


class TestMultilabelClassification:
    def test_trainer(self, tmp_path: Path, accelerator_config: dict) -> None:
        """A multilabel run trains, tests and saves per-label predictions."""
        model = DeterministicClassification(
            MLP(n_inputs=2, n_outputs=3, n_hidden=[8]),
            nn.BCEWithLogitsLoss(),
            task="multilabel",
        )
        datamodule = ToyMultilabelDataModule()
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))

        trainer.fit(model, datamodule)
        trainer.test(model, datamodule=datamodule)

        df = pd.read_csv(os.path.join(tmp_path, model.pred_file_name))
        assert "pred" not in df.columns
        for i in range(3):
            assert set(df[f"pred_{i}"].unique()) <= {0, 1}
            assert f"class_prob_{i}" in df.columns
            assert f"target_{i}" in df.columns
