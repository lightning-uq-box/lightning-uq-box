# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Epinet tests for behaviour the config sweeps cannot reach.

``tests/configs/{regression,classification,image_classification}/epinet.yaml`` are in
the sweeps of ``test_regression.py``, ``test_classification.py`` and
``test_image_classification.py``, which already cover fit, test and the prediction
file for both the MLP and the convolutional base. What is left here is behaviour a
one-epoch smoke run cannot see: the stop-gradient on the captured features, freezing
the base *head* as well as its body (unlike ``freeze_model_backbone``, which is why
epinet is absent from ``frozen_config_paths``), the shared-index joint prediction
contract, Gaussian bootstrapping, which needs the stable ``batch["index"]`` IDs that no
toy datamodule provides, and the guards that turn a silently wrong result into an error.
"""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import torch
from conftest import minimal_trainer_kwargs
from lightning import Trainer
from torch import Tensor, nn

from lightning_uq_box.datamodules import TwoMoonsDataModule
from lightning_uq_box.models import MLP, ConvEnsemblePriorFunction
from lightning_uq_box.uq_methods import (
    EpinetClassification,
    EpinetGaussianNoiseLoss,
    EpinetRegression,
)

Task = Literal["regression", "multiclass", "binary", "multilabel"]


def _epinet_model(
    task: Task = "regression", n_outputs: int | None = None, **kwargs: Any
) -> EpinetRegression | EpinetClassification:
    """Build a small epinet over an MLP base network.

    Args:
        task: ``"regression"`` builds an :class:`EpinetRegression`, the others an
            :class:`EpinetClassification` with that task
        n_outputs: width of the base network output, defaulting to one per task
        **kwargs: forwarded to the model, overriding the small test defaults

    Returns:
        the constructed epinet model
    """
    defaults: dict[str, Any] = {
        "index_dim": 4,
        "num_index_samples": 2,
        "num_pred_samples": 5,
        "epinet_hidden_dims": [8],
        "prior_hidden_dims": [4],
    } | kwargs
    if task == "regression":
        base = MLP(n_inputs=1, n_hidden=[16], n_outputs=n_outputs or 1)
        return EpinetRegression(base, nn.MSELoss(), **defaults)
    n_outputs = n_outputs or (3 if task == "multilabel" else 2)
    loss: nn.Module = (
        nn.BCEWithLogitsLoss()
        if task == "multilabel" or n_outputs == 1
        else nn.CrossEntropyLoss()
    )
    base = MLP(n_inputs=2, n_hidden=[16], n_outputs=n_outputs)
    return EpinetClassification(base, loss, task=task, **defaults)


def _conv_base(n_outputs: int = 4) -> nn.Module:
    """Conv backbone with a linear head, the shape of every ResNet."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, n_outputs),
    )


INDEX_SHAPE = r"\[num_samples, index_dim\]"
IMAGE_INPUT = "not supported for image"


def _predict_with_index(indices: Tensor) -> Tensor:
    """Draw prediction samples with an explicitly supplied index tensor."""
    return _epinet_model().eval().predict_samples(torch.randn(3, 1), indices)


def _bootstrap_loss(data_index: Tensor | None = None) -> object:
    """Score a bootstrap model in training mode, with the given example IDs."""
    model = _epinet_model(bootstrap_noise_scale=0.2, num_train=10)
    return model.compute_loss(torch.randn(4, 1), torch.randn(4, 1), data_index)


def _wrap(base: nn.Module, **kwargs: Any) -> object:
    """Wrap a base network in an epinet, for guards on the base network itself."""
    return EpinetClassification(base, nn.CrossEntropyLoss(), **kwargs)


class _OutFeaturesOnly(nn.Module):
    """Advertises out_features but no input width, so the head cannot be sized.

    A module can advertise ``out_features`` -- which is what selects it as the output
    layer -- while exposing no input width at all.
    """

    def __init__(self, out_features: int = 4) -> None:
        super().__init__()
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        return x


class _ExtraAxis(nn.Module):
    """Returns [batch_size, num_outputs, 1], outside the vector output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).unsqueeze(-1)


# The epinet wraps an arbitrary base network, so most misuse is only detectable at
# construction or on the first forward. Each guard turns a silently wrong result -- a
# broadcast index, a pooled feature map read as a vector, a prior of the wrong width --
# into an explicit error, so every one of them is pinned here.
#
# Two notes on the less obvious rows. The flattened input size of an image network is
# not recoverable from the module tree, and sizing the default prior from the classifier
# head's in_features instead produced a prior of the wrong width that only failed later,
# inside its own forward pass, with an opaque matrix-shape error. Image inputs are
# detected from the first parameterized layer, which breaks on a Conv2d or a Linear, so
# a conv body with a linear head and a conv-only network are different traversals and
# both are covered.
GUARDS: list[tuple[str, Callable[[], object], str]] = [
    ("index_dim", lambda: _epinet_model(index_dim=0), "must be positive"),
    (
        "num_index_samples",
        lambda: _epinet_model(num_index_samples=0),
        "must be positive",
    ),
    ("num_pred_samples", lambda: _epinet_model(num_pred_samples=1), "at least 2"),
    ("negative_bootstrap", lambda: _epinet_model(bootstrap_noise_scale=-0.1), "nonneg"),
    (
        "bootstrap_num_train",
        lambda: _epinet_model(bootstrap_noise_scale=0.2),
        "num_train",
    ),
    # equation 9 assumes a scalar target, so a multi-output base is refused
    (
        "bootstrap_vector_output",
        lambda: _epinet_model(n_outputs=2, bootstrap_noise_scale=0.2, num_train=10),
        "scalar regression output",
    ),
    ("binary_logits", lambda: _epinet_model("binary", 3), "one or two output logits"),
    (
        "conv_needs_prior",
        lambda: _wrap(_conv_base(), use_input_features=False, input_prior_scale=0.3),
        "cannot be inferred",
    ),
    (
        "conv_head_features",
        lambda: _wrap(_conv_base(), use_input_features=True),
        IMAGE_INPUT,
    ),
    (
        "conv_only_features",
        lambda: _wrap(nn.Sequential(nn.Conv2d(3, 2, 3)), use_input_features=True),
        IMAGE_INPUT,
    ),
    (
        "no_input_width",
        lambda: _wrap(nn.Sequential(nn.Linear(2, 4), _OutFeaturesOnly())),
        "neither in_features nor in_channels",
    ),
    (
        "non_vector_output",
        lambda: EpinetRegression(
            _ExtraAxis(), nn.MSELoss(), index_dim=2, epinet_hidden_dims=[4]
        ).compute_loss(torch.randn(4, 2), torch.randn(4, 3)),
        r"\[batch_size, num_outputs\]",
    ),
    ("index_ndim", lambda: _predict_with_index(torch.randn(4)), INDEX_SHAPE),
    ("index_width", lambda: _predict_with_index(torch.randn(2, 7)), INDEX_SHAPE),
    ("index_empty", lambda: _predict_with_index(torch.randn(0, 4)), INDEX_SHAPE),
    ("ids_missing", _bootstrap_loss, "stable example IDs"),
    ("ids_not_integer", lambda: _bootstrap_loss(torch.zeros(4)), "one integer ID"),
    (
        "ids_out_of_range",
        lambda: _bootstrap_loss(torch.tensor([0, 1, 2, 10])),
        r"\[0, num_train\)",
    ),
]


@pytest.mark.parametrize(
    "call,match",
    [(call, match) for _, call, match in GUARDS],
    ids=[name for name, _, _ in GUARDS],
)
def test_guards(call: Callable[[], object], match: str) -> None:
    """Reject the arguments, base networks and inputs the epinet cannot use."""
    with pytest.raises(ValueError, match=match):
        call()


class TestFeatureCaptureAndStopGradient:
    @pytest.mark.parametrize("use_input_features", [False, True])
    def test_features_are_captured_without_touching_the_base(
        self, use_input_features: bool
    ) -> None:
        """A forward pre-hook reads the output layer's input, changing nothing."""
        base = MLP(n_inputs=1, n_hidden=[16], n_outputs=1)
        before = copy.deepcopy(base.state_dict())
        model = EpinetRegression(
            base, nn.MSELoss(), index_dim=4, use_input_features=use_input_features
        )

        after = base.state_dict()
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), f"{key} changed"

        X = torch.randn(6, 1)
        model.model(X)
        features = model.extract_features(X)
        # the MLP's output layer takes the 16-wide hidden representation
        assert features.shape == (6, 17 if use_input_features else 16)
        if use_input_features:
            assert torch.allclose(features[:, 16:], X)

    def test_reading_features_before_a_forward_raises(self) -> None:
        model = _epinet_model()
        with pytest.raises(RuntimeError, match="No features were captured"):
            model.extract_features(torch.randn(3, 1))

    def test_epinet_loss_leaves_the_base_untouched_but_the_full_loss_does_not(
        self,
    ) -> None:
        """sg[phi(x)] keeps an epinet-only loss off the base; mu(x) still trains it."""
        model = _epinet_model(freeze_backbone=False)
        assert all(p.requires_grad for p in model.model.parameters())

        X = torch.randn(8, 1)
        model.zero_grad()
        model.model(X)
        features = model.extract_features(X)
        index = model.sample_index(8, X.device)
        model.epinet(features, X.detach(), index).pow(2).sum().backward()
        for name, param in model.model.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0.0), (
                f"base parameter {name} received a gradient through the epinet, "
                "so the stop-gradient is missing"
            )

        model.zero_grad()
        loss, _, _ = model.compute_loss(torch.randn(8, 1), torch.randn(8, 1))
        loss.backward()
        grads = [p.grad for p in model.model.parameters() if p.grad is not None]
        assert grads, "base network received no gradients at all"
        assert any(torch.any(g != 0.0) for g in grads)

    def test_spatial_features_are_pooled_to_a_vector(self) -> None:
        """A conv head with no Flatten captures [B, C, H, W], which is mean-pooled.

        Pooling rather than flattening keeps the epinet head's width tied to the
        channel count and independent of the input size.
        """
        base = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1)
        )
        model = EpinetClassification(
            base,
            nn.CrossEntropyLoss(),
            index_dim=4,
            epinet_hidden_dims=[8],
            input_prior_scale=0.0,
            use_input_features=False,
        )
        x = torch.randn(2, 3, 8, 8)
        # Run the base network directly so the hook captures the feature map; the
        # epinet's own forward would reject this base for not returning a vector.
        model.model(x)
        captured = model._features
        assert captured is not None
        features = model.extract_features(x)
        assert features.shape == (2, 8), "expected mean over the spatial axes"
        torch.testing.assert_close(features, captured.flatten(2).mean(-1))


class TestFreezeBackbone:
    """Replaces a ``frozen_config_paths`` entry.

    ``TestFrozenBackbone`` in the task sweeps asserts the base network's *head* stays
    trainable, which is the behaviour of ``freeze_model_backbone``. The epinet freezes
    the base head too, so it needs its own assertions.
    """

    @pytest.mark.parametrize("task", ["regression", "multiclass"])
    def test_the_whole_base_is_frozen_but_the_epinet_is_not(self, task: Task) -> None:
        model = _epinet_model(task, freeze_backbone=True, epi_prior_scale=1.0)
        for name, param in model.model.named_parameters():
            assert not param.requires_grad, (
                f"base parameter {name} is trainable; the epinet freezes the whole "
                "base network, its head included"
            )
        assert all(p.requires_grad for p in model.epinet.train_epinet.parameters())
        assert all(not p.requires_grad for p in model.epinet.epi_prior.parameters())

    def test_optimizer_only_sees_trainable_parameters(self) -> None:
        """Frozen priors and a frozen base must not collect updates or weight decay."""
        model = _epinet_model(
            freeze_backbone=True, epi_prior_scale=1.0, input_prior_scale=0.3
        )
        config = cast(dict[str, Any], model.configure_optimizers())
        in_optimizer = {
            id(p) for group in config["optimizer"].param_groups for p in group["params"]
        }
        assert in_optimizer == {id(p) for p in model.epinet.train_epinet.parameters()}

    def test_frozen_base_is_unchanged_by_a_real_fit(
        self, tmp_path: Path, accelerator_config: dict
    ) -> None:
        """The paper's headline use case: train the epinet on a fixed base."""
        model = _epinet_model("multiclass", freeze_backbone=True)
        before = copy.deepcopy(model.model.state_dict())
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))
        trainer.fit(model, TwoMoonsDataModule(batch_size=16))
        for key, value in model.model.state_dict().items():
            assert torch.equal(before[key], value), (
                f"frozen base parameter {key} changed during training"
            )

    def test_an_unfrozen_base_does_change(self) -> None:
        """The control for the test above, which would also pass for a dead model."""
        model = _epinet_model("multiclass", freeze_backbone=False)
        before = copy.deepcopy(model.model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(2):
            optimizer.zero_grad()
            loss, _, _ = model.compute_loss(torch.randn(8, 2), torch.randint(2, (8,)))
            loss.backward()
            optimizer.step()
        after = model.model.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_frozen_batchnorm_and_dropout_stay_fixed(self) -> None:
        """A frozen base must not drift through running statistics or dropout."""
        base = nn.Sequential(
            nn.Linear(2, 8), nn.BatchNorm1d(8), nn.Dropout(0.8), nn.Linear(8, 2)
        )
        model = EpinetClassification(base, nn.CrossEntropyLoss(), freeze_backbone=True)
        x, y = torch.randn(6, 2), torch.randint(2, (6,))
        before = copy.deepcopy(base.state_dict())
        expected = base(x).detach().clone()

        model.train()
        model.compute_loss(x, y)[0].backward()
        assert not base.training
        assert model.epinet.train_epinet.training
        assert not model.epinet.epi_prior.training
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(base(x), expected, rtol=0, atol=0)


# (task, n_outputs, expected keys, expected pred/uct/samples shapes at batch size 6)
PREDICTION_CONTRACTS: list[tuple[Task, int, set[str], tuple[tuple[int, ...], ...]]] = [
    (
        "regression",
        1,
        {"pred", "pred_uct", "epistemic_uct", "samples"},
        ((6, 1), (6,), (6, 1, 5)),
    ),
    ("multiclass", 2, {"pred", "pred_uct", "logits"}, ((6, 2), (6,), (6, 2, 5))),
    ("binary", 1, {"pred", "pred_uct", "logits"}, ((6, 2), (6,), (6, 2, 5))),
    ("binary", 2, {"pred", "pred_uct", "logits"}, ((6, 2), (6,), (6, 2, 5))),
    ("multilabel", 3, {"pred", "pred_uct", "logits"}, ((6, 3), (6,), (6, 3, 5))),
]


class TestPredictionContract:
    @pytest.mark.parametrize(
        "task,n_outputs,expected_keys,shapes",
        PREDICTION_CONTRACTS,
        ids=[f"{task}-{n}" for task, n, _, _ in PREDICTION_CONTRACTS],
    )
    def test_predict_step_keys_and_shapes(
        self,
        task: Task,
        n_outputs: int,
        expected_keys: set[str],
        shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        """Every task returns the documented keys, shapes and sample layout."""
        model = _epinet_model(task, n_outputs=n_outputs).eval()
        out = model.predict_step(torch.randn(6, 1 if task == "regression" else 2))
        assert set(out) == expected_keys

        samples_key = "samples" if task == "regression" else "logits"
        assert out["pred"].shape == shapes[0]
        assert out["pred_uct"].shape == shapes[1]
        assert out[samples_key].shape == shapes[2]
        # each slice of the last axis is one coherent index sample, so they differ
        assert not torch.allclose(out[samples_key][..., 0], out[samples_key][..., 1])
        assert torch.all(out["pred_uct"] > 0.0)
        if task in ("multiclass", "binary"):
            torch.testing.assert_close(out["pred"].sum(-1), torch.ones(6))

    def test_a_single_binary_logit_is_zero_padded(self) -> None:
        """One logit is padded to two columns, so pred is the mean sigmoid."""
        model = _epinet_model("binary", n_outputs=1).eval()
        out = model.predict_step(torch.randn(6, 2))
        torch.testing.assert_close(
            out["pred"][:, 1], out["logits"][:, 1].sigmoid().mean(-1)
        )

    def test_multilabel_probabilities_stay_independent(self) -> None:
        """Multilabel outputs are sigmoid, not softmax, so they need not sum to one."""
        model = _epinet_model("multilabel").eval()
        out = model.predict_step(torch.randn(6, 2))
        torch.testing.assert_close(out["pred"], out["logits"].sigmoid().mean(-1))

    @pytest.mark.parametrize("n_outputs", [1, 2])
    def test_binary_targets_are_adapted_for_metrics(self, n_outputs: int) -> None:
        """Binary targets are normalized for both the loss and the metrics.

        A single-logit model is trained against float targets but scored against
        integer class labels, so ``test_step`` normalizes them for the CSV contract.
        """
        model = _epinet_model("binary", n_outputs=n_outputs)
        x, y = torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 1, 0])
        with patch.object(model, "log"):
            loss = model.training_step({"input": x, "target": y}, 0)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(model.train_metrics.compute()["trainAcc"])

        model.eval()
        target = y[:, None].float() if n_outputs == 1 else y
        result = model.test_step({"input": x, "target": target}, 0)
        assert result["pred"].shape == (6, 2)
        torch.testing.assert_close(result["pred"].sum(-1), torch.ones(6))

    def test_zero_index_recovers_the_base_network(self) -> None:
        """The epinet is linear in z, so z = 0 leaves only mu(x)."""
        model = _epinet_model(epi_prior_scale=1.0, input_prior_scale=0.3)
        X = torch.randn(6, 1)
        assert torch.allclose(
            model.forward(X, torch.zeros(6, 4)), model.model(X), atol=1e-6
        )

    def test_an_omitted_index_is_sampled_per_example(self) -> None:
        model = _epinet_model()
        X = torch.randn(6, 1)
        out = model.forward(X)
        assert out.shape == model.forward(X, model.sample_index(6, X.device)).shape
        assert torch.isfinite(out).all()

    def test_conv_base_with_an_explicit_conv_prior(self) -> None:
        """An image prior passed as input_prior, the CIFAR-10 configuration's shape."""
        model = EpinetClassification(
            _conv_base(),
            nn.CrossEntropyLoss(),
            index_dim=4,
            num_pred_samples=3,
            use_input_features=False,
            input_prior_scale=1.0,
            input_prior=ConvEnsemblePriorFunction(
                in_channels=3, n_outputs=4, num_ensemble=4, input_size=32
            ),
        )
        out = model.eval().predict_step(torch.randn(5, 3, 32, 32))
        assert out["pred"].shape == (5, 4)
        assert out["logits"].shape == (5, 4, 3)


class TestJointPredictionContract:
    def test_shared_indices_agree_across_batches(self) -> None:
        """Joint predictions need one index set reused across batches, not resampled."""
        model = _epinet_model().eval()
        x = torch.randn(12, 1)
        z = model.sample_index(7, x.device)
        whole = model.predict_samples(x, z)
        chunks = torch.cat([model.predict_samples(chunk, z) for chunk in x.split(4)])
        torch.testing.assert_close(chunks, whole)
        for k in range(len(z)):
            torch.testing.assert_close(whole[..., k], model(x, z[k].expand(len(x), -1)))

    def test_internally_sampled_indices_follow_the_model_dtype(self) -> None:
        model = _epinet_model().double()
        x = torch.randn(4, 1, dtype=torch.float64)
        assert model(x).dtype == torch.float64
        assert model.predict_samples(x).dtype == torch.float64

    def test_the_capture_hook_belongs_to_a_deepcopy(self) -> None:
        """A hook closing over the original would write into it from the copy."""
        original = _epinet_model()
        cloned = copy.deepcopy(original)
        cloned(torch.randn(3, 1))
        assert cloned._features is not None
        assert original._features is None

    def test_the_base_runs_once_per_batch(self) -> None:
        """Repeating the batch for K indices must not rerun the base K times."""
        model = _epinet_model()
        calls: list[int] = []
        handle = model.model.register_forward_hook(lambda *args: calls.append(1))
        model.compute_loss(torch.randn(4, 1), torch.randn(4, 1))
        assert len(calls) == 1
        model.predict_samples(torch.randn(4, 1))
        assert len(calls) == 2
        handle.remove()

    def test_the_logged_batch_size_accounts_for_the_index_repeat(self) -> None:
        """Training repeats the batch K times, which the logged batch size must match."""
        model = _epinet_model(num_index_samples=3)
        with patch.object(model, "log") as log:
            model.training_step(
                {"input": torch.randn(8, 1), "target": torch.randn(8, 1)}, 0
            )
        assert log.call_args.kwargs["batch_size"] == 8 * 3


class TestGaussianBootstrap:
    def test_loss_matches_equation_nine(self) -> None:
        loss = EpinetGaussianNoiseLoss(0.5)
        preds, targets = torch.tensor([[2.0], [3.0]]), torch.tensor([[1.0], [2.0]])
        c = torch.eye(2)
        z = torch.tensor([[2.0, 4.0], [2.0, 4.0]])
        assert loss(preds, targets, c, z).item() == pytest.approx(0.5)

    def test_signatures_and_priors_survive_a_checkpoint(self, tmp_path: Path) -> None:
        """Signatures and priors are buffers and submodules, so they are restored.

        The reloaded model is built with a different ``prior_seed``, so identical
        predictions prove the priors came from the checkpoint rather than being
        re-randomized on load.
        """
        model = _epinet_model(
            bootstrap_noise_scale=0.2,
            num_train=10,
            epi_prior_scale=1.0,
            input_prior_scale=0.3,
        )
        # a registered buffer is typed as the union Tensor | Module, so narrow it
        signatures = cast(Tensor, model.bootstrap_signatures).clone()
        torch.testing.assert_close(signatures.norm(dim=-1), torch.ones(10))

        # a buffer registered as a parameter would drift once gradients flow
        model.compute_loss(torch.randn(4, 1), torch.randn(4, 1), torch.arange(4))[
            0
        ].backward()
        torch.testing.assert_close(model.bootstrap_signatures, signatures)

        path = tmp_path / "bootstrap.ckpt"
        trainer = Trainer(default_root_dir=str(tmp_path), logger=False)
        trainer.strategy.connect(model)
        trainer.save_checkpoint(path)
        loaded = EpinetRegression.load_from_checkpoint(
            path,
            model=MLP(n_inputs=1, n_hidden=[16], n_outputs=1),
            loss_fn=nn.MSELoss(),
            prior_seed=999,
        )

        x, z = torch.randn(4, 1), torch.randn(4, 4)
        torch.testing.assert_close(loaded(x, z), model(x, z))
        torch.testing.assert_close(loaded.bootstrap_signatures, signatures)
        assert loaded.bootstrap_noise_scale == 0.2

    def test_eval_mode_drops_the_bootstrap_term(self) -> None:
        """Outside training the plain loss is used on unperturbed targets, with no IDs."""
        model = _epinet_model(bootstrap_noise_scale=0.2, num_train=10).eval()
        loss, out, target = model.compute_loss(torch.randn(4, 1), torch.randn(4, 1))
        torch.testing.assert_close(loss, nn.MSELoss()(out, target))

    def test_the_loss_respects_estimator_major_ids(self) -> None:
        """A batch-major repeat would pair the wrong signature with the wrong index."""
        model = _epinet_model(bootstrap_noise_scale=0.2, num_train=10)
        x, y, ids = torch.randn(3, 1), torch.randn(3, 1), torch.tensor([9, 1, 9])
        z = torch.randn(2, 4)
        with patch.object(model, "sample_index", return_value=z):
            loss, out, target = model.compute_loss(x, y, ids)
        c = cast(Tensor, model.bootstrap_signatures)[ids].repeat(2, 1)
        noise = 0.2 * (c * z.repeat_interleave(3, dim=0)).sum(-1, keepdim=True)
        torch.testing.assert_close(loss, (out - target - noise).square().mean())


def test_configure_optimizers_wires_up_a_scheduler() -> None:
    """A supplied scheduler is returned in the lr dict and monitors val_loss."""
    from functools import partial

    model = _epinet_model(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=1),
    )
    # Lightning types configure_optimizers as a wide union whose TypedDict arm has no
    # lr_scheduler key, so cast to the plain dict this method actually returns.
    config = cast(dict[str, Any], model.configure_optimizers())
    scheduler_config = config["lr_scheduler"]
    assert scheduler_config["monitor"] == "val_loss"
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.StepLR)


@pytest.mark.slow
def test_linear_gaussian_epinet_matches_bayesian_posterior() -> None:
    """Optimize equation 9 using exact Gaussian second moments (linear head)."""
    torch.manual_seed(5)
    n, dim, sigma = 16, 128, 0.3
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 0.7 * x + sigma * torch.randn_like(x)
    model = EpinetRegression(
        nn.Linear(1, 1, bias=False),
        nn.MSELoss(),
        index_dim=dim,
        num_index_samples=2 * dim,
        epinet_hidden_dims=[],
        use_input_features=False,
        epinet_concat_index=False,
        input_prior_scale=0,
        epi_prior_scale=1,
        bootstrap_noise_scale=sigma,
        num_train=n,
    )
    head = model.epinet.train_epinet.mlp[0]
    prior = model.epinet.epi_prior.mlp[0]
    assert isinstance(head, nn.Linear) and isinstance(prior, nn.Linear)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        prior.weight.normal_(std=dim**-0.5)
        prior.bias.zero_()
    head.bias.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(
        params, max_iter=50, tolerance_grad=1e-9, tolerance_change=1e-12
    )
    z = torch.cat([torch.eye(dim), -torch.eye(dim)]) * dim**0.5

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        with patch.object(model, "sample_index", return_value=z):
            loss = model.compute_loss(x, y, torch.arange(n))[0]
        loss = loss + sigma**2 / n * sum(p.square().sum() for p in params)
        loss.backward()
        return loss

    optimizer.step(closure)
    samples = model.eval().predict_samples(torch.ones(1, 1), z).flatten()
    expected_mean = (x * y).sum() / (x.square().sum() + sigma**2)
    expected_std = (sigma**2 / (x.square().sum() + sigma**2)).sqrt()
    torch.testing.assert_close(samples.mean(), expected_mean, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        samples.std(correction=0), expected_std, atol=0.01, rtol=0.2
    )
