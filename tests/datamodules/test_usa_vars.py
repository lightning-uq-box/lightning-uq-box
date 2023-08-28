import os

import pytest
from _pytest.fixtures import SubRequest

from lightning_uq_box.datamodules import USAVarsFeatureExtractedDataModule


class TestUSAVarsFeatureExtractedDataModule:
    @pytest.fixture
    def datamodule(self, request: SubRequest) -> USAVarsFeatureExtractedDataModule:
        pytest.importorskip("pandas", minversion="1.1.3")
        root = os.path.join("tests", "data", "usa_vars")
        batch_size = 1
        num_workers = 0

        dm = USAVarsFeatureExtractedDataModule(
            root=root, batch_size=batch_size, num_workers=num_workers
        )
        dm.prepare_data()
        return dm

    def test_train_dataloader(
        self, datamodule: USAVarsFeatureExtractedDataModule
    ) -> None:
        datamodule.setup("fit")
        assert len(datamodule.train_dataloader()) == 3
        batch = next(iter(datamodule.train_dataloader()))
        assert batch["image"].shape[0] == datamodule.batch_size

    def test_val_dataloader(
        self, datamodule: USAVarsFeatureExtractedDataModule
    ) -> None:
        datamodule.setup("validate")
        assert len(datamodule.val_dataloader()) == 3
        batch = next(iter(datamodule.val_dataloader()))
        assert batch["image"].shape[0] == datamodule.batch_size

    def test_test_dataloader(
        self, datamodule: USAVarsFeatureExtractedDataModule
    ) -> None:
        datamodule.setup("test")
        assert len(datamodule.test_dataloader()) == 3
        batch = next(iter(datamodule.test_dataloader()))
        assert batch["image"].shape[0] == datamodule.batch_size
