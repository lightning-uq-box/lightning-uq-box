# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Contract tests for canonical task-aware base methods."""

import lightning
import pytest
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lightning_uq_box.uq_methods import (
    NLL,
    ClassificationTask,
    Deterministic,
    MCDropout,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
)


class SingleBatchDataset(Dataset[dict[str, torch.Tensor]]):
    """A one-item dataset that returns an already collated tutorial batch."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        """Store one complete batch."""
        self.batch = batch

    def __len__(self) -> int:
        """Return the number of complete batches."""
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the stored batch."""
        assert index == 0
        return self.batch


def test_canonical_runtime_has_no_state_dict_entries() -> None:
    """Metrics/runtime remain run state rather than checkpoint model state."""
    module = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    assert not module.task_runtime.state_dict()
    assert not any(key.startswith("task_runtime.") for key in module.state_dict())


def test_equivalent_canonical_module_loads_strictly() -> None:
    """Task runtime registration does not add incompatible checkpoint keys."""
    source = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    target = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    incompatible = target.load_state_dict(source.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_direct_checkpoint_load_is_strict_when_dependencies_are_supplied(
    tmp_path,
) -> None:
    """The canonical checkpoint boundary documents caller-supplied modules."""
    source = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    checkpoint_path = tmp_path / "canonical.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "hyper_parameters": dict(source.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint_path,
    )
    restored = Deterministic.load_from_checkpoint(
        checkpoint_path, model=nn.Linear(2, 1), loss_fn=nn.MSELoss(), strict=True
    )
    assert restored.task == RegressionTask()


def test_binary_one_logit_keeps_batch_dimension_for_one_example() -> None:
    """A single-item batch is a batch in the public prediction contract."""
    module = Deterministic(
        nn.Linear(2, 1),
        nn.BCEWithLogitsLoss(),
        task=ClassificationTask(mode="binary", binary_encoding="one_logit"),
    )
    output = module.predict_step(torch.zeros(1, 2))
    assert output["pred"].shape == (1, 1)
    assert output["pred_uct"].shape == (1,)
    assert output["logits"].shape == (1, 1)


def test_canonical_mc_dropout_keeps_its_sample_axis_method_owned() -> None:
    """MC Dropout exposes sampled logits while the runtime stays stateless."""
    model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(), nn.Linear(4, 1))
    module = MCDropout(
        model,
        num_mc_samples=3,
        loss_fn=nn.BCEWithLogitsLoss(),
        task=ClassificationTask(mode="binary", binary_encoding="one_logit"),
    )
    output = module.predict_step(torch.zeros(1, 2))
    assert output["pred"].shape == (1, 1)
    assert output["logits"].shape == (1, 1, 3)
    assert not module.task_runtime.state_dict()


def test_canonical_tutorial_paths_train_and_test(tmp_path) -> None:
    """The deterministic and Gaussian MC Dropout tutorial APIs run end to end."""
    classification = Deterministic(
        nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2)),
        nn.CrossEntropyLoss(),
        task=ClassificationTask(mode="multiclass"),
    )
    classification_batch = {
        "input": torch.zeros(2, 2),
        "target": torch.zeros(2, dtype=torch.long),
    }
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmp_path / "classification",
        logger=False,
        enable_model_summary=False,
    )
    classification_loader = DataLoader(
        SingleBatchDataset(classification_batch), batch_size=None
    )
    trainer.fit(classification, train_dataloaders=classification_loader)
    trainer.test(classification, dataloaders=classification_loader)

    regression = MCDropout(
        nn.Sequential(nn.Linear(1, 4), nn.Dropout(), nn.Linear(4, 2)),
        num_mc_samples=2,
        loss_fn=NLL(),
        task=RegressionTask(),
        prediction_kind="gaussian",
    )
    regression_batch = {"input": torch.zeros(2, 1), "target": torch.zeros(2, 1)}
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmp_path / "regression",
        logger=False,
        enable_model_summary=False,
    )
    regression_loader = DataLoader(
        SingleBatchDataset(regression_batch), batch_size=None
    )
    trainer.fit(regression, train_dataloaders=regression_loader)
    trainer.test(regression, dataloaders=regression_loader)


@pytest.mark.parametrize(
    ("task", "model", "loss", "input_tensor", "pred_shape"),
    [
        (RegressionTask(), nn.Linear(2, 1), nn.MSELoss(), torch.zeros(2, 2), (2, 1)),
        (
            ClassificationTask(mode="multiclass"),
            nn.Linear(2, 3),
            nn.CrossEntropyLoss(),
            torch.zeros(2, 2),
            (2, 3),
        ),
        (
            SegmentationTask(mode="multiclass"),
            nn.Conv2d(1, 3, kernel_size=1),
            nn.CrossEntropyLoss(),
            torch.zeros(2, 1, 4, 4),
            (2, 3, 4, 4),
        ),
        (
            PixelRegressionTask(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.MSELoss(),
            torch.zeros(2, 1, 4, 4),
            (2, 1, 4, 4),
        ),
    ],
)
def test_declared_deterministic_capabilities_have_contract_payloads(
    task, model, loss, input_tensor, pred_shape
) -> None:
    """Each task family returned by the canonical pilot is explicitly shaped."""
    module = Deterministic(model, loss, task=task)
    payload = module.predict_step(input_tensor)
    assert payload["pred"].shape == pred_shape
    module.output_schema.validate_payload(payload)
