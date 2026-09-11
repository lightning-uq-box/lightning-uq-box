# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test the Epinet method."""

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning import Trainer
from torch import nn

from lightning_uq_box.datamodules import (
    ToyHeteroscedasticDatamodule,
    TwoMoonsDataModule,
)
from lightning_uq_box.models import MLP, ConvEnsemblePriorFunction, Epinet, ProjectedMLP
from lightning_uq_box.uq_methods import EpinetClassification, EpinetRegression


def build_regression_model(**kwargs: Any) -> EpinetRegression:
    """Build a small EpinetRegression for testing."""
    defaults: dict[str, Any] = {
        "index_dim": 4,
        "num_index_samples": 2,
        "num_pred_samples": 5,
        "epinet_hidden_dims": [8],
        "prior_hidden_dims": [4],
    }
    defaults.update(kwargs)
    return EpinetRegression(
        MLP(n_inputs=1, n_hidden=[16], n_outputs=1), nn.MSELoss(), **defaults
    )


def build_classification_model(**kwargs: Any) -> EpinetClassification:
    """Build a small EpinetClassification for testing."""
    defaults: dict[str, Any] = {
        "index_dim": 4,
        "num_index_samples": 2,
        "num_pred_samples": 5,
        "epinet_hidden_dims": [8],
        "prior_hidden_dims": [4],
    }
    defaults.update(kwargs)
    return EpinetClassification(
        MLP(n_inputs=2, n_hidden=[16], n_outputs=2), nn.CrossEntropyLoss(), **defaults
    )


class TestProjectedMLP:
    def test_einsum_orientation_against_hand_computation(self) -> None:
        """Pin the reshape orientation with known weights and one-hot indices.

        A transposed ``[batch, index_dim, n_outputs]`` reshape still runs and still
        trains, so only an explicit hand-computed case rules it out. The final linear
        layer is set to emit ``[0, 1, ..., 7]`` regardless of input; reshaping that to
        ``[n_outputs=2, index_dim=4]`` gives rows ``[0, 1, 2, 3]`` and ``[4, 5, 6, 7]``,
        so contracting with the i-th one-hot index must return ``[i, i + 4]``.
        """
        model = ProjectedMLP(
            in_features=2, hidden_dims=[3], n_outputs=2, index_dim=4, concat_index=False
        )
        final = model.mlp[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.arange(8, dtype=torch.float))

        features = torch.zeros(1, 2)
        for i in range(4):
            index = torch.zeros(1, 4)
            index[0, i] = 1.0
            out = model(features, index)
            expected = torch.tensor([[float(i), float(i + 4)]])
            assert torch.allclose(out, expected), (
                f"one-hot index {i} gave {out.tolist()}, expected {expected.tolist()}. "
                "The reshape in ProjectedMLP.forward is likely transposed."
            )

    def test_linear_in_the_index(self) -> None:
        """The projection is linear in z, so scaling z scales the output."""
        model = ProjectedMLP(4, [8], 2, 3, concat_index=False)
        features = torch.randn(5, 4)
        index = torch.randn(5, 3)
        assert torch.allclose(
            model(features, 2.0 * index), 2.0 * model(features, index), atol=1e-5
        )

    def test_output_shape(self) -> None:
        model = ProjectedMLP(4, [8, 8], 3, 6)
        out = model(torch.randn(7, 4), torch.randn(7, 6))
        assert out.shape == (7, 3)


class TestFeatureCapture:
    def test_hook_captures_expected_width(self) -> None:
        """The pre-hook stashes the input of the base network's output layer."""
        model = build_regression_model(use_input_features=False)
        X = torch.randn(6, 1)
        model.model(X)
        features = model.extract_features(X)
        # the MLP's output layer takes the 16-wide hidden representation
        assert features.shape == (6, 16)

    def test_input_features_are_concatenated(self) -> None:
        model = build_regression_model(use_input_features=True)
        X = torch.randn(6, 1)
        model.model(X)
        features = model.extract_features(X)
        assert features.shape == (6, 17)
        assert torch.allclose(features[:, 16:], X)

    def test_base_network_is_left_untouched(self) -> None:
        """Attaching an epinet must not change the base network's parameters."""
        base = MLP(n_inputs=1, n_hidden=[16], n_outputs=1)
        before = copy.deepcopy(base.state_dict())
        EpinetRegression(base, nn.MSELoss(), index_dim=4)
        after = base.state_dict()
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), f"{key} changed"

    def test_base_module_structure_unchanged(self) -> None:
        """No layer is replaced, so the module tree is identical."""
        base = MLP(n_inputs=1, n_hidden=[16], n_outputs=1)
        names_before = [name for name, _ in base.named_modules()]
        EpinetRegression(base, nn.MSELoss(), index_dim=4)
        assert [name for name, _ in base.named_modules()] == names_before

    def test_reading_features_before_forward_raises(self) -> None:
        model = build_regression_model()
        with pytest.raises(RuntimeError, match="No features were captured"):
            model.extract_features(torch.randn(3, 1))


class TestStopGradient:
    def test_epinet_loss_leaves_base_grads_at_zero(self) -> None:
        """sg[phi(x)] means an epinet-only loss never reaches the base network."""
        model = build_regression_model(freeze_backbone=False)
        assert all(p.requires_grad for p in model.model.parameters())

        X = torch.randn(8, 1)
        model.zero_grad()
        model.model(X)
        features = model.extract_features(X)
        index = model.sample_index(8, X.device)
        loss = model.epinet(features, X.detach(), index).pow(2).sum()
        loss.backward()

        for name, param in model.model.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0.0), (
                f"base parameter {name} received a gradient through the epinet, "
                "so the stop-gradient is missing"
            )

    def test_full_loss_does_reach_the_base_network(self) -> None:
        """The base network still learns through its own output term."""
        model = build_regression_model(freeze_backbone=False)
        model.zero_grad()
        loss, _, _ = model.compute_loss(torch.randn(8, 1), torch.randn(8, 1))
        loss.backward()
        grads = [p.grad for p in model.model.parameters() if p.grad is not None]
        assert grads, "base network received no gradients at all"
        assert any(torch.any(g != 0.0) for g in grads)


class TestFrozenPrior:
    @pytest.mark.parametrize("input_prior_scale", [0.0, 0.3])
    def test_prior_parameters_do_not_require_grad(
        self, input_prior_scale: float
    ) -> None:
        model = build_regression_model(
            epi_prior_scale=1.0, input_prior_scale=input_prior_scale
        )
        assert all(not p.requires_grad for p in model.epinet.epi_prior.parameters())
        if model.epinet.input_prior is not None:
            assert all(
                not p.requires_grad for p in model.epinet.input_prior.parameters()
            )
        assert all(p.requires_grad for p in model.epinet.train_epinet.parameters())

    def test_priors_are_in_the_state_dict(self) -> None:
        """The prior must be checkpointed, not re-randomized on reload."""
        model = build_regression_model(epi_prior_scale=1.0, input_prior_scale=0.3)
        keys = model.state_dict().keys()
        assert any(k.startswith("epinet.epi_prior.") for k in keys)
        assert any(k.startswith("epinet.input_prior.") for k in keys)

    def test_checkpoint_round_trip_reproduces_predictions(self, tmp_path: Path) -> None:
        """A saved and reloaded model gives identical outputs for a fixed index."""
        model = build_regression_model(epi_prior_scale=1.0, input_prior_scale=0.3)
        X = torch.randn(5, 1)
        index = torch.randn(5, 4)
        expected = model.forward(X, index)

        path = tmp_path / "epinet.ckpt"
        trainer = Trainer(default_root_dir=str(tmp_path))
        trainer.strategy.connect(model)
        trainer.save_checkpoint(path)

        reloaded = build_regression_model(
            epi_prior_scale=1.0, input_prior_scale=0.3, prior_seed=999
        )
        state = torch.load(path, weights_only=False)["state_dict"]
        reloaded.load_state_dict(state)

        assert torch.allclose(reloaded.forward(X, index), expected, atol=1e-6)

    def test_prior_moves_with_the_module(self) -> None:
        """Registered submodules follow .to(), unlike a plain attribute."""
        model = build_regression_model(epi_prior_scale=1.0, input_prior_scale=0.3)
        model = model.to(torch.float64)
        input_prior = model.epinet.input_prior
        assert input_prior is not None
        assert next(model.epinet.epi_prior.parameters()).dtype == torch.float64
        assert next(input_prior.parameters()).dtype == torch.float64


class TestIndexSensitivity:
    def test_different_indices_give_different_outputs(self) -> None:
        model = build_regression_model()
        X = torch.randn(6, 1)
        out_a = model.forward(X, torch.zeros(6, 4))
        out_b = model.forward(X, torch.ones(6, 4))
        assert not torch.allclose(out_a, out_b)

    def test_the_same_index_is_deterministic(self) -> None:
        model = build_regression_model()
        X = torch.randn(6, 1)
        index = torch.randn(6, 4)
        assert torch.allclose(model.forward(X, index), model.forward(X, index))

    def test_zero_index_recovers_the_base_network(self) -> None:
        """The epinet is linear in z, so z = 0 leaves only mu(x)."""
        model = build_regression_model(epi_prior_scale=1.0, input_prior_scale=0.3)
        X = torch.randn(6, 1)
        assert torch.allclose(
            model.forward(X, torch.zeros(6, 4)), model.model(X), atol=1e-6
        )

    def test_predict_step_gives_non_degenerate_uncertainty(self) -> None:
        model = build_regression_model(num_pred_samples=32)
        out = model.predict_step(torch.randn(10, 1))
        assert torch.all(out["pred_uct"] > 0.0)

    def test_prior_scales_drive_the_spread(self) -> None:
        """With both priors off and an untrained epinet, spread is much smaller."""
        torch.manual_seed(0)
        with_prior = build_regression_model(
            num_pred_samples=64, epi_prior_scale=0.0, input_prior_scale=5.0
        )
        torch.manual_seed(0)
        without_prior = build_regression_model(
            num_pred_samples=64, epi_prior_scale=0.0, input_prior_scale=0.0
        )
        X = torch.randn(16, 1)
        assert (
            with_prior.predict_step(X)["pred_uct"].mean()
            > without_prior.predict_step(X)["pred_uct"].mean()
        )


class TestFreezeBackbone:
    """Replaces a frozen_config_paths entry.

    ``TestFrozenBackbone`` asserts the base network's *head* stays trainable, which is
    the behaviour of ``freeze_model_backbone``. The epinet freezes the base head too,
    so it needs its own assertion.
    """

    @pytest.mark.parametrize(
        "builder", [build_regression_model, build_classification_model]
    )
    def test_every_base_parameter_is_frozen(self, builder) -> None:
        model = builder(freeze_backbone=True)
        for name, param in model.model.named_parameters():
            assert not param.requires_grad, (
                f"base parameter {name} is trainable; the epinet freezes the whole "
                "base network, its head included"
            )

    @pytest.mark.parametrize(
        "builder", [build_regression_model, build_classification_model]
    )
    def test_non_prior_epinet_parameters_stay_trainable(self, builder) -> None:
        model = builder(freeze_backbone=True, epi_prior_scale=1.0)
        assert all(p.requires_grad for p in model.epinet.train_epinet.parameters())
        assert all(not p.requires_grad for p in model.epinet.epi_prior.parameters())

    def test_optimizer_only_sees_trainable_parameters(self) -> None:
        model = build_regression_model(
            freeze_backbone=True, epi_prior_scale=1.0, input_prior_scale=0.3
        )
        config = model.configure_optimizers()
        assert isinstance(config, dict)
        optimizer = config["optimizer"]
        in_optimizer = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        expected = {id(p) for p in model.epinet.train_epinet.parameters()}
        assert in_optimizer == expected

    def test_frozen_base_is_unchanged_by_training(self, tmp_path: Path) -> None:
        """The paper's headline use case: train the epinet on a fixed base."""
        model = build_classification_model(freeze_backbone=True)
        before = copy.deepcopy(model.model.state_dict())

        datamodule = TwoMoonsDataModule(batch_size=16)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, datamodule)

        after = model.model.state_dict()
        for key in before:
            assert torch.equal(before[key], after[key]), (
                f"frozen base parameter {key} changed during training"
            )

    def test_unfrozen_base_does_change(self, tmp_path: Path) -> None:
        model = build_classification_model(freeze_backbone=False)
        before = copy.deepcopy(model.model.state_dict())

        datamodule = TwoMoonsDataModule(batch_size=16)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, datamodule)

        after = model.model.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)


class TestPredictStep:
    def test_regression_keys_and_shapes(self) -> None:
        model = build_regression_model(num_pred_samples=5)
        out = model.predict_step(torch.randn(9, 1))
        assert set(out) == {"pred", "pred_uct", "epistemic_uct", "samples"}
        assert out["pred"].shape == (9, 1)
        assert out["pred_uct"].shape == (9,)
        assert out["epistemic_uct"].shape == (9,)
        assert out["samples"].shape == (9, 1, 5)

    def test_classification_keys_and_shapes(self) -> None:
        model = build_classification_model(num_pred_samples=5)
        out = model.predict_step(torch.randn(9, 2))
        assert set(out) == {"pred", "pred_uct", "logits"}
        assert out["pred"].shape == (9, 2)
        assert out["pred_uct"].shape == (9,)
        assert out["logits"].shape == (9, 2, 5)
        assert torch.allclose(out["pred"].sum(-1), torch.ones(9), atol=1e-5)

    def test_sample_layout_is_sample_major_on_the_last_axis(self) -> None:
        """Each slice on the last axis must be one coherent index sample."""
        model = build_regression_model(num_pred_samples=8)
        X = torch.randn(4, 1)
        samples = model.predict_samples(X)
        assert samples.shape == (4, 1, 8)
        # different samples genuinely differ
        assert not torch.allclose(samples[..., 0], samples[..., 1])


class TestTrainingLoop:
    @pytest.mark.parametrize("freeze_backbone", [False, True])
    def test_regression_fit_and_test(
        self, tmp_path: Path, freeze_backbone: bool
    ) -> None:
        model = build_regression_model(freeze_backbone=freeze_backbone)
        datamodule = ToyHeteroscedasticDatamodule(batch_size=16)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            limit_test_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        assert (tmp_path / model.pred_file_name).exists()

    def test_classification_fit_and_test(self, tmp_path: Path) -> None:
        model = build_classification_model()
        datamodule = TwoMoonsDataModule(batch_size=16)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            limit_test_batches=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
        )
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        assert (tmp_path / model.pred_file_name).exists()

    def test_training_step_batch_size_accounts_for_index_repeat(self) -> None:
        """The logged batch size must reflect the K-fold repeat."""
        model = build_regression_model(num_index_samples=3)
        logged: dict[str, int] = {}

        def fake_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = kwargs.get("batch_size", -1)

        # ty: intentionally swapping the logger out for a recorder
        model.log = fake_log  # ty: ignore[invalid-assignment]
        batch = {"input": torch.randn(8, 1), "target": torch.randn(8, 1)}
        model.training_step(batch, 0)
        assert logged["train_loss"] == 8 * 3


class TestEpinetModule:
    def test_input_prior_skipped_when_scale_is_zero(self) -> None:
        epinet = Epinet(
            n_feature_inputs=4, n_raw_inputs=2, n_outputs=2, input_prior_scale=0.0
        )
        assert epinet.input_prior is None

    def test_conv_backbone_rejects_input_features(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 2, 3))
        with pytest.raises(ValueError, match="not supported for image inputs"):
            EpinetClassification(model, nn.CrossEntropyLoss(), use_input_features=True)


class TestConvEnsemblePriorFunction:
    """The image prior, as used by the paper's CIFAR-10 configuration."""

    def test_output_shape_and_frozen(self) -> None:
        prior = ConvEnsemblePriorFunction(
            in_channels=3, n_outputs=10, num_ensemble=4, input_size=32
        )
        out = prior(torch.randn(6, 3, 32, 32), torch.randn(6, 4))
        assert out.shape == (6, 10)
        assert all(not p.requires_grad for p in prior.parameters())

    def test_linear_in_the_index(self) -> None:
        prior = ConvEnsemblePriorFunction(
            in_channels=3, n_outputs=5, num_ensemble=3, input_size=32
        )
        x = torch.randn(2, 3, 32, 32)
        index = torch.randn(2, 3)
        assert torch.allclose(prior(x, 2.0 * index), 2.0 * prior(x, index), atol=1e-4)

    def test_members_are_independently_initialized(self) -> None:
        prior = ConvEnsemblePriorFunction(
            in_channels=3, n_outputs=5, num_ensemble=2, input_size=32
        )
        first, second = prior.members[0], prior.members[1]
        assert isinstance(first, nn.Sequential) and isinstance(second, nn.Sequential)
        first_conv, second_conv = first[0], second[0]
        assert isinstance(first_conv, nn.Conv2d) and isinstance(second_conv, nn.Conv2d)
        assert not torch.equal(first_conv.weight, second_conv.weight)

    def test_seed_makes_it_reproducible(self) -> None:
        a = ConvEnsemblePriorFunction(3, 5, 2, input_size=32, seed=7)
        b = ConvEnsemblePriorFunction(3, 5, 2, input_size=32, seed=7)
        x = torch.randn(2, 3, 32, 32)
        index = torch.randn(2, 2)
        assert torch.allclose(a(x, index), b(x, index))

    def test_rejects_input_too_small_for_the_conv_stack(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            ConvEnsemblePriorFunction(3, 5, 2, input_size=8)


class TestCustomInputPrior:
    """Passing an explicit prior module, the CIFAR-10 configuration's shape."""

    def test_conv_prior_is_used_and_frozen(self) -> None:
        conv_prior = ConvEnsemblePriorFunction(
            in_channels=3, n_outputs=10, num_ensemble=4, input_size=32
        )
        epinet = Epinet(
            n_feature_inputs=16,
            n_raw_inputs=0,
            n_outputs=10,
            index_dim=4,
            epi_prior_scale=4.0,
            input_prior_scale=1.0,
            input_prior=conv_prior,
        )
        assert epinet.input_prior is conv_prior
        assert all(not p.requires_grad for p in epinet.input_prior.parameters())

        out = epinet(torch.randn(2, 16), torch.randn(2, 3, 32, 32), torch.randn(2, 4))
        assert out.shape == (2, 10)
