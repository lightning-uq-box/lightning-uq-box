# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Contract tests for canonical task-aware base methods."""

from dataclasses import dataclass
from typing import Any, ClassVar

import lightning
import pytest
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

import lightning_uq_box.uq_methods.task_runtime as task_runtime_module
from lightning_uq_box.uq_methods import (
    NLL,
    ClassificationTask,
    Deterministic,
    MCDropout,
    PixelRegressionTask,
    RegressionTask,
    SegmentationTask,
    TaskMethodBase,
    TaskSpec,
)
from lightning_uq_box.uq_methods.method_specs import (
    BINARY_ONE_LOGIT_SCHEMA,
    CLASSIFICATION_SCHEMA,
    PIXEL_REGRESSION_SCHEMA,
    REGRESSION_SCHEMA,
    SEGMENTATION_SCHEMA,
)
from lightning_uq_box.uq_methods.task_runtime import TaskRuntime


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


@dataclass(frozen=True, kw_only=True)
class UnsupportedTask(TaskSpec):
    """A task value intentionally absent from the runtime dispatch table."""

    class_path: ClassVar[str] = "tests.UnsupportedTask"


class TupleOutputModel(nn.Module):
    """Model fixture whose output intentionally violates canonical contracts."""

    def __init__(self) -> None:
        """Build a model with both a dropout module and a linear output layer."""
        super().__init__()
        self.dropout = nn.Dropout()
        self.output = nn.Linear(2, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        """Return a non-Tensor output to exercise public contract errors."""
        return (self.output(self.dropout(X)),)


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


def test_task_runtime_properties_target_rules_and_metric_lifecycle() -> None:
    """The runtime owns exact target semantics and run-only metric state."""
    regression = TaskRuntime(RegressionTask(), REGRESSION_SCHEMA, num_outputs=1)
    assert regression.train_metrics is not None
    assert regression.val_metrics is not None
    assert regression.test_metrics is not None
    prediction = torch.zeros(1, 1)
    target = torch.ones(1, 1)
    regression.update_metrics("train", prediction, target)
    assert regression.compute_and_reset("train")

    multiclass = TaskRuntime(
        ClassificationTask(mode="multiclass"), CLASSIFICATION_SCHEMA, num_outputs=2
    )
    class_target = torch.tensor([[1], [0]])
    assert multiclass.normalize_target(class_target).shape == (2,)
    assert multiclass.target_for_loss(class_target, torch.zeros(2, 2)).shape == (2,)

    binary = TaskRuntime(
        ClassificationTask(mode="binary", binary_encoding="one_logit"),
        BINARY_ONE_LOGIT_SCHEMA,
        num_outputs=1,
    )
    binary_target = torch.tensor([[1.0], [0.0]])
    assert binary.normalize_target(binary_target).shape == (2,)
    assert binary.target_for_loss(
        torch.tensor([1.0, 0.0]), torch.zeros(2, 1)
    ).shape == (2, 1)


def test_task_runtime_rejects_unknown_task_types() -> None:
    """A task requires an explicit metrics implementation rather than fallback."""
    schema = REGRESSION_SCHEMA.__class__(
        task_type=UnsupportedTask,
        raw_axes=("batch", "target"),
        metric_input="raw_prediction",
        fields=REGRESSION_SCHEMA.fields,
    )
    with pytest.raises(TypeError, match="Unsupported task runtime"):
        TaskRuntime(UnsupportedTask(), schema, num_outputs=1)


def test_task_runtime_copies_results_and_creates_dense_directory(tmp_path) -> None:
    """Runtime results preserve tensor ownership and dense paths are explicit."""
    runtime = TaskRuntime(RegressionTask(), REGRESSION_SCHEMA, num_outputs=1)
    payload = {"pred": torch.ones(1, 1)}
    batch: dict[str, Any] = {
        "input": torch.zeros(1, 1),
        "target": torch.zeros(1, 1),
        "index": torch.tensor([4]),
        "metadata": "keep-me",
    }
    result = runtime.test_result(payload, batch, input_key="input", target_key="target")
    assert result["pred"].data_ptr() != payload["pred"].data_ptr()
    assert result["index"].data_ptr() != batch["index"].data_ptr()
    assert result["metadata"] == "keep-me"
    with pytest.raises(TypeError, match="must be a Tensor"):
        runtime.test_result(
            payload,
            {"input": torch.zeros(1, 1), "target": "not-a-tensor"},
            input_key="input",
            target_key="target",
        )
    assert runtime.on_test_start(str(tmp_path), save_predictions=True) is None

    dense = TaskRuntime(
        SegmentationTask(mode="multiclass"), SEGMENTATION_SCHEMA, num_outputs=2
    )
    assert dense.on_test_start(str(tmp_path), save_predictions=False) is None
    pred_dir = dense.on_test_start(str(tmp_path), save_predictions=True)
    assert pred_dir is not None
    assert (tmp_path / "preds").is_dir()


def test_task_runtime_writer_dispatch_is_non_mutating(tmp_path, monkeypatch) -> None:
    """Each task family dispatches its writer with a copied result dictionary."""
    calls: list[tuple[str, dict[str, Any], tuple[Any, ...], dict[str, Any]]] = []

    def record(name: str):
        def writer(outputs, *args, **kwargs) -> None:
            outputs["pred"].zero_()
            calls.append((name, outputs, args, kwargs))

        return writer

    monkeypatch.setattr(
        task_runtime_module, "save_regression_predictions", record("csv")
    )
    monkeypatch.setattr(
        task_runtime_module, "save_classification_predictions", record("class")
    )
    monkeypatch.setattr(task_runtime_module, "save_image_predictions", record("image"))
    outputs: dict[str, Any] = {"pred": torch.ones(1, 1), "target": torch.zeros(1, 1)}

    regression = TaskRuntime(RegressionTask(), REGRESSION_SCHEMA, num_outputs=1)
    regression.write_test_result(
        outputs,
        root_dir=str(tmp_path),
        batch_idx=0,
        save_predictions=False,
        prediction_dir=None,
    )
    classification = TaskRuntime(
        ClassificationTask(mode="binary", binary_encoding="one_logit"),
        BINARY_ONE_LOGIT_SCHEMA,
        num_outputs=1,
    )
    classification.write_test_result(
        outputs,
        root_dir=str(tmp_path),
        batch_idx=1,
        save_predictions=False,
        prediction_dir=None,
    )
    pixel = TaskRuntime(PixelRegressionTask(), PIXEL_REGRESSION_SCHEMA, num_outputs=1)
    pixel.write_test_result(
        outputs,
        root_dir=str(tmp_path),
        batch_idx=2,
        save_predictions=False,
        prediction_dir=None,
    )
    pixel.write_test_result(
        outputs,
        root_dir=str(tmp_path),
        batch_idx=3,
        save_predictions=True,
        prediction_dir=str(tmp_path / "preds"),
    )
    pixel.write_test_result(
        None,  # type: ignore[arg-type]
        root_dir=str(tmp_path),
        batch_idx=4,
        save_predictions=True,
        prediction_dir=str(tmp_path / "preds"),
    )

    assert [call[0] for call in calls] == ["csv", "class", "image"]
    assert calls[1][3] == {"task": "binary", "binary_encoding": "one_logit"}
    assert torch.equal(outputs["pred"], torch.ones(1, 1))


def test_task_method_base_requires_spec_and_exposes_runtime_services() -> None:
    """Canonical bases fail closed until their concrete method declares a spec."""
    incomplete = TaskMethodBase()
    with pytest.raises(TypeError, match="Task runtime"):
        incomplete.save_task_hyperparameters({})
    incomplete.model = nn.Linear(2, 1)
    with pytest.raises(TypeError, match="MethodSpec"):
        incomplete.initialize_task_runtime(None, default_task=RegressionTask())

    module = Deterministic(nn.Linear(2, 1), nn.MSELoss(), task=RegressionTask())
    assert module.supports_task(RegressionTask())
    assert not module.supports_task(UnsupportedTask())
    assert module.train_metrics is module.task_runtime.train_metrics
    assert module.val_metrics is module.task_runtime.val_metrics
    assert module.test_metrics is module.task_runtime.test_metrics


def test_deterministic_validates_output_contract_and_lifecycle_paths(
    monkeypatch,
) -> None:
    """Canonical deterministic methods test conversion, validation, and schedulers."""
    invalid_binary = Deterministic(
        nn.Linear(2, 2),
        nn.BCEWithLogitsLoss(),
        task=ClassificationTask(mode="binary", binary_encoding="one_logit"),
    )
    with pytest.raises(ValueError, match="one_logit"):
        invalid_binary.predict_step(torch.zeros(1, 2))

    multilabel = Deterministic(
        nn.Linear(2, 3),
        nn.BCEWithLogitsLoss(),
        task=ClassificationTask(mode="multilabel"),
    )
    payload = multilabel.prediction_payload(torch.zeros(1, 3))
    assert payload["pred"].shape == (1, 3)
    assert payload["pred_uct"].shape == (1,)

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    module = Deterministic(
        model,
        nn.MSELoss(),
        task=RegressionTask(),
        freeze_backbone=True,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 1),
    )
    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    batch = {"input": torch.zeros(1, 2), "target": torch.zeros(1, 1)}
    assert module.validation_step(batch, 0).shape == ()
    module.on_validation_epoch_end()
    optimizer_config = module.configure_optimizers()
    assert isinstance(optimizer_config, dict)
    assert "lr_scheduler" in optimizer_config

    tuple_module = Deterministic(
        TupleOutputModel(), nn.MSELoss(), task=RegressionTask()
    )
    with pytest.raises(TypeError, match="Tensor model output"):
        tuple_module.training_step(batch, 0)
    with pytest.raises(TypeError, match="Tensor model output"):
        tuple_module.predict_step(torch.zeros(1, 2))
    with pytest.raises(TypeError, match="Tensor model output"):
        tuple_module.test_step(batch, 0)


def test_canonical_mc_dropout_covers_sampling_and_explicit_conversions(
    monkeypatch,
) -> None:
    """MC Dropout keeps every sampling and distribution branch method-owned."""
    model = nn.Sequential(
        nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(), nn.Linear(2, 1)
    )
    module = MCDropout(
        model,
        num_mc_samples=2,
        loss_fn=nn.MSELoss(),
        task=RegressionTask(),
        dropout_layer_names=["2"],
    )
    module.eval()
    module.activate_dropout()
    assert not model[1].training
    assert model[2].training
    point_payload = module.prediction_payload(torch.zeros(2, 1, 1))
    assert point_payload["pred"].shape == (1, 1)
    assert point_payload["epistemic_uct"].shape == (1, 1)

    with pytest.raises(ValueError, match="at least one"):
        MCDropout(nn.Linear(2, 1), 0, nn.MSELoss())
    with pytest.raises(ValueError, match="prediction_kind"):
        MCDropout(nn.Linear(2, 1), 1, nn.MSELoss(), prediction_kind="unknown")
    no_dropout = MCDropout(nn.Linear(2, 1), 1, nn.MSELoss())
    with pytest.raises(UserWarning, match="No dropout"):
        no_dropout.activate_dropout()
    tuple_module = MCDropout(TupleOutputModel(), 1, nn.MSELoss())
    with pytest.raises(TypeError, match="Tensor model outputs"):
        tuple_module._samples(torch.zeros(1, 2))
    tuple_burnin = MCDropout(
        TupleOutputModel(), 1, nn.MSELoss(), task=RegressionTask(), burnin_epochs=1
    )
    monkeypatch.setattr(tuple_burnin, "log", lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match="Tensor model output"):
        tuple_burnin.training_step(
            {"input": torch.zeros(1, 2), "target": torch.zeros(1, 1)}, 0
        )

    gaussian = MCDropout(
        nn.Sequential(nn.Linear(2, 2), nn.Dropout()),
        2,
        NLL(),
        task=RegressionTask(),
        prediction_kind="gaussian",
    )
    with pytest.raises(ValueError, match="two-channel"):
        gaussian.prediction_payload(torch.zeros(2, 1, 1))
    with pytest.raises(ValueError, match="two-channel"):
        gaussian.metric_prediction(torch.zeros(1, 1), "test")
    assert gaussian.metric_prediction(torch.zeros(1, 2), "test").shape == (1, 1)

    for task, channels in [
        (ClassificationTask(mode="multilabel"), 3),
        (ClassificationTask(mode="binary", binary_encoding="one_logit"), 1),
        (ClassificationTask(mode="multiclass"), 3),
    ]:
        classifier = MCDropout(
            nn.Sequential(nn.Linear(2, channels), nn.Dropout()),
            2,
            nn.BCEWithLogitsLoss() if channels == 1 else nn.CrossEntropyLoss(),
            task=task,
        )
        payload = classifier.prediction_payload(torch.zeros(2, 1, channels))
        assert payload["pred"].shape == (1, channels)
        assert payload["pred_uct"].shape == (1,)
        assert payload["logits"].shape == (1, channels, 2)
        assert classifier.metric_prediction(torch.zeros(1, channels), "test").shape in {
            (1, channels),
            (1,),
        }

    burnin = MCDropout(
        nn.Sequential(nn.Linear(2, 1), nn.Dropout()),
        2,
        nn.L1Loss(),
        task=RegressionTask(),
        burnin_epochs=1,
    )
    monkeypatch.setattr(burnin, "log", lambda *args, **kwargs: None)
    loss = burnin.training_step(
        {"input": torch.zeros(1, 2), "target": torch.zeros(1, 1)}, 0
    )
    assert loss.shape == ()
