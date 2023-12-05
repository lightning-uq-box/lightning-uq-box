import builtins
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

import lightning_uq_box
from lightning_uq_box.datasets import USAVarsFeatureExtracted

pytest.importorskip("pandas", minversion="1.1.3")


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestUSAVarsFeatureExtracted:
    @pytest.fixture(
        params=zip(
            ["train", "val", "test"],
            [
                ["elevation", "population", "treecover"],
                ["elevation", "population"],
                ["treecover"],
            ],
            ["rcf", "resnet18"],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> USAVarsFeatureExtracted:
        split, labels, feature_extractor = request.param

        monkeypatch.setattr(
            lightning_uq_box.datasets.usa_vars, "download_url", download_url
        )
        data_url = os.path.join(
            "tests", "data", "usa_vars", f"{feature_extractor}_usa_vars.csv"
        )
        monkeypatch.setattr(USAVarsFeatureExtracted, "data_url", data_url)

        root = str(tmp_path)
        return USAVarsFeatureExtracted(
            root, split, labels, feature_extractor, download=True
        )

    def test_getitem(self, dataset: USAVarsFeatureExtracted) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].ndim == 1
        assert len(x.keys()) == 4  # features, labels, centroid_lat, centroid_lon
        assert x["image"].shape[0] == 512  # 512 features
        assert len(dataset.labels) == len(x["labels"])
        assert len(x["centroid_lat"]) == 1
        assert len(x["centroid_lon"]) == 1

    def test_len(self, dataset: USAVarsFeatureExtracted) -> None:
        if dataset.split == "train":
            assert len(dataset) == 3
        elif dataset.split == "val":
            assert len(dataset) == 3
        else:
            assert len(dataset) == 3

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "usa_vars", "rcf_usa_vars.csv")
        root = str(tmp_path)
        shutil.copy(pathname, root)
        USAVarsFeatureExtracted(root, feature_extractor="rcf")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            USAVarsFeatureExtracted(str(tmp_path))

    @pytest.fixture(params=["pandas"])
    def mock_missing_module(self, monkeypatch: MonkeyPatch, request: SubRequest) -> str:
        import_orig = builtins.__import__
        package = str(request.param)

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        return package

    def test_mock_missing_module(
        self, dataset: USAVarsFeatureExtracted, mock_missing_module: str
    ) -> None:
        package = mock_missing_module
        if package == "pandas":
            with pytest.raises(
                ImportError,
                match=f"{package} is not installed and is required to use this dataset",
            ):
                USAVarsFeatureExtracted(
                    dataset.root, feature_extractor=dataset.feature_extractor
                )
