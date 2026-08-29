# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""DKL regression tests for checkpoint round-tripping.

The config-driven smoke tests in test_classification.py/test_regression.py
train a DKL model and call ``trainer.test(...)`` on the *same in-memory
object*, so they never exercise loading a checkpoint into a fresh model --
which is how the bug covered here survived: ``gp_layer``, ``likelihood``,
``scale_to_bounds`` and ``elbo_fn`` are all created lazily by
``_build_model()``, which runs from ``configure_optimizers`` and therefore only
during ``fit``. Restoring a checkpoint skips that path entirely, so every
GP/likelihood/kernel tensor was rejected as an unexpected key and both
``load_from_checkpoint`` and ``Trainer.test(ckpt_path=...)`` raised
``RuntimeError`` for a model that had trained perfectly well.
"""

import numpy as np
import torch
from conftest import minimal_trainer_kwargs
from lightning import LightningDataModule, Trainer
from torch import nn
from torch.utils.data import DataLoader

from lightning_uq_box.uq_methods import DKLClassification, DKLRegression
from lightning_uq_box.uq_methods.deep_kernel_learning import _stack_targets

NUM_CLASSES = 4
NUM_FEATURES = 16


class _TinyBackbone(nn.Module):
    """Small feature extractor so the GP layer has something to sit on."""

    def __init__(self, num_outputs: int = NUM_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ToyDataModule(LightningDataModule):
    """Deterministic in-memory dataset in the dict format DKL expects."""

    def __init__(self, n: int = 64, regression: bool = False) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8, generator=generator)
        self.y = (
            torch.randn(n, 1, generator=generator)
            if regression
            else torch.randint(0, NUM_CLASSES, (n,), generator=generator)
        )

    def _loader(self) -> DataLoader:
        samples = [
            {"input": self.x[i], "target": self.y[i]} for i in range(len(self.y))
        ]

        def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
            return {
                "input": torch.stack([b["input"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
            }

        return DataLoader(samples, batch_size=16, collate_fn=collate)

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()

    def test_dataloader(self) -> DataLoader:
        return self._loader()

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int = 0) -> dict:
        return batch


class TestDKLCheckpointRoundTrip:
    def test_classification_load_from_checkpoint(
        self, accelerator_config, tmp_path
    ) -> None:
        """A DKL classifier must reload with its GP weights bit-exact.

        Regression test: this used to raise ``RuntimeError`` listing every
        ``gp_layer.*`` / ``likelihood.*`` tensor as an unexpected key.
        """
        datamodule = _ToyDataModule()
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=2)
        )
        trainer.fit(model, datamodule)

        ckpt_path = tmp_path / "dkl_classification.ckpt"
        trainer.save_checkpoint(ckpt_path)

        reloaded = DKLClassification.load_from_checkpoint(
            ckpt_path,
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )

        original_state = model.state_dict()
        reloaded_state = reloaded.state_dict()
        assert set(original_state) == set(reloaded_state)
        for key, value in original_state.items():
            if value.is_floating_point():
                assert torch.allclose(value, reloaded_state[key].to(value.device)), key

        # The GP posterior itself must match. predict_step draws 64 random
        # likelihood samples, so compare the underlying distribution rather
        # than the sampled probabilities.
        model.eval()
        reloaded.eval()
        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            expected = model.forward(inputs)
            actual = reloaded.forward(inputs.to(reloaded.device))
        assert torch.allclose(expected.mean, actual.mean.to(expected.mean.device))
        assert torch.allclose(expected.stddev, actual.stddev.to(expected.stddev.device))

    def test_classification_test_with_ckpt_path(
        self, accelerator_config, tmp_path
    ) -> None:
        """``Trainer.test(ckpt_path=...)`` must run and score the trained model.

        This is the evaluate-a-finished-run path: a fresh model plus a
        checkpoint, which is how a training job's results are scored. It used
        to raise ``RuntimeError`` outright.

        Metrics are deliberately *not* compared for equality here. Both go
        through ``predict_step``, which draws 64 random likelihood samples, so
        they are stochastic: repeating ``test()`` on one unchanged model varies
        ``test_loss`` by ~0.03 on this toy setup. Bit-exact weight restoration
        is asserted in ``test_classification_load_from_checkpoint`` instead;
        what matters here is that the run completes and lands in the same
        region rather than at the ~1.39 = ln(4) of an untrained GP.
        """
        datamodule = _ToyDataModule()
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=2)
        )
        trainer.fit(model, datamodule)
        ckpt_path = tmp_path / "dkl_test_path.ckpt"
        trainer.save_checkpoint(ckpt_path)
        expected = trainer.test(model, datamodule, verbose=False)[0]

        fresh = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        actual = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path)).test(
            fresh, datamodule, ckpt_path=str(ckpt_path), verbose=False
        )[0]

        assert "testAcc" in actual
        # Loose bound: well inside the sampling noise measured for repeated
        # test() calls on one model, but far tighter than the gap that a
        # failed restore would produce (an unbuilt GP would not load at all,
        # and a wrong n_train_points shifts the KL term by orders of
        # magnitude).
        assert abs(actual["test_loss"] - expected["test_loss"]) < 0.1

    def test_n_train_points_survives_checkpoint(
        self, accelerator_config, tmp_path
    ) -> None:
        """``num_data`` scales the ELBO's KL term, so it must round-trip.

        Without it the reloaded model would silently normalize the KL term by
        1 instead of the training-set size, changing every reported test_loss.
        """
        datamodule = _ToyDataModule()
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        )
        trainer.fit(model, datamodule)
        ckpt_path = tmp_path / "dkl_num_data.ckpt"
        trainer.save_checkpoint(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert checkpoint["n_train_points"] == len(datamodule.y)

        reloaded = DKLClassification.load_from_checkpoint(
            ckpt_path,
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        assert reloaded.n_train_points == len(datamodule.y)
        assert reloaded.elbo_fn.num_data == len(datamodule.y)

    def test_regression_load_from_checkpoint(
        self, accelerator_config, tmp_path
    ) -> None:
        """The same round-trip must hold for DKLRegression."""
        datamodule = _ToyDataModule(regression=True)
        model = DKLRegression(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_targets=1,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=2)
        )
        trainer.fit(model, datamodule)
        ckpt_path = tmp_path / "dkl_regression.ckpt"
        trainer.save_checkpoint(ckpt_path)

        reloaded = DKLRegression.load_from_checkpoint(
            ckpt_path,
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_targets=1,
            gp_kernel="RBF",
        )
        model.eval()
        reloaded.eval()
        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            expected = model.forward(inputs)
            actual = reloaded.forward(inputs.to(reloaded.device))
        assert torch.allclose(expected.mean, actual.mean.to(expected.mean.device))


class TestStackTargets:
    """`compute_initial_values` must accept datasets whose labels are not tensors.

    Standard torchvision classification datasets return plain Python ints
    (``CIFAR10[0]`` is ``(Tensor, int)``), which made
    ``torch.stack([train_dataset[j][1] ...])`` raise
    ``TypeError: expected Tensor as element 0 in argument 0, but got int``
    during ``configure_optimizers`` -- so DKL could not be trained on CIFAR-10
    or any similar dataset at all.
    """

    def test_int_labels(self) -> None:
        """Python ints are the torchvision classification case."""
        stacked = _stack_targets([1, 2, 3])
        assert stacked.tolist() == [1, 2, 3]

    def test_numpy_scalars(self) -> None:
        """Some datasets hand back numpy scalars instead."""
        assert _stack_targets([np.int64(7), np.int64(8)]).tolist() == [7, 8]

    def test_tensor_targets_unchanged(self) -> None:
        """Tensor targets must keep working exactly as before."""
        stacked = _stack_targets([torch.tensor([1.0]), torch.tensor([2.0])])
        assert stacked.shape == (2, 1)
        assert stacked.dtype == torch.float32

    def test_torchvision_style_dataset_trains(
        self, accelerator_config, tmp_path
    ) -> None:
        """End-to-end: a (Tensor, int) dataset must get through fit().

        This is the shape of the failure that killed the first CIFAR-10 smoke
        run -- it surfaces from configure_optimizers, before step 1.
        """

        class _TupleIntLabelDataset(torch.utils.data.Dataset):
            # compute_initial_values samples up to 1000 points and splits them
            # with .chunk(10), so the dataset must be large enough to yield 10
            # non-empty chunks.
            def __init__(self, n: int = 64) -> None:
                generator = torch.Generator().manual_seed(0)
                self.x = torch.randn(n, 3, 8, 8, generator=generator)
                self.y = torch.randint(
                    0, NUM_CLASSES, (n,), generator=generator
                ).tolist()

            def __len__(self) -> int:
                return len(self.y)

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
                # int, not a tensor -- exactly what torchvision returns.
                return self.x[idx], self.y[idx]

        class _DM(LightningDataModule):
            def __init__(self) -> None:
                super().__init__()
                self.dataset = _TupleIntLabelDataset()

            def _loader(self) -> DataLoader:
                def collate(batch: list) -> dict[str, torch.Tensor]:
                    images, labels = zip(*batch)
                    return {
                        "input": torch.stack(images),
                        "target": torch.tensor(labels, dtype=torch.long),
                    }

                return DataLoader(self.dataset, batch_size=16, collate_fn=collate)

            def train_dataloader(self) -> DataLoader:
                return self._loader()

            def val_dataloader(self) -> DataLoader:
                return self._loader()

            def test_dataloader(self) -> DataLoader:
                return self._loader()

            def on_after_batch_transfer(self, batch: dict, dataloader_idx: int = 0):
                return batch

        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        )
        trainer.fit(model, _DM())
        assert model.dkl_model_built


class TestScaleFeatures:
    """The forward pass must match the reference DUE implementation.

    DUE's ``DKL.forward`` (due/dkl.py) is exactly
    ``self.gp(self.feature_extractor(x))``. ``DKLBase`` used to insert a
    ``ScaleToBounds`` between the two unconditionally, which broke training:
    ``compute_initial_values`` fits the lengthscale to *unscaled* features, so
    rescaling afterwards invalidates it and pushes the RBF kernel into
    saturation, where it passes almost no gradient back to the backbone.
    """

    def test_default_is_due_faithful(self) -> None:
        """Scaling must be off by default, as in DUE."""
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        assert model.scale_features is False

    def test_forward_matches_due(self, accelerator_config, tmp_path) -> None:
        """With scaling off, forward() must equal gp(feature_extractor(x)).

        Compares against DUE's wrapper driving the *same* GP object, so any
        difference is attributable to the forward path rather than to
        separately initialized GP state.
        """
        datamodule = _ToyDataModule()
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
        )
        Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        ).fit(model, datamodule)
        model.eval()

        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            actual = model.forward(inputs)
            # DUE's DKL.forward, transcribed.
            expected = model.gp_layer(model.feature_extractor(inputs))

        assert torch.allclose(actual.mean, expected.mean)
        assert torch.allclose(actual.stddev, expected.stddev)

    def test_scale_features_opt_in_changes_forward(
        self, accelerator_config, tmp_path
    ) -> None:
        """The legacy behaviour must remain reachable for old checkpoints."""
        datamodule = _ToyDataModule()
        model = DKLClassification(
            feature_extractor=_TinyBackbone(),
            n_inducing_points=8,
            num_classes=NUM_CLASSES,
            gp_kernel="RBF",
            scale_features=True,
        )
        Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=1)
        ).fit(model, datamodule)
        model.eval()

        assert model.scale_features is True
        inputs = datamodule.x[:8].to(model.device)
        with torch.no_grad():
            scaled = model.forward(inputs)
            unscaled = model.gp_layer(model.feature_extractor(inputs))
        # The whole point of the flag is that it changes the GP input.
        assert not torch.allclose(scaled.mean, unscaled.mean)

    def test_scaling_distorts_distances_relative_to_fitted_lengthscale(self) -> None:
        """Scaling breaks the lengthscale that compute_initial_values fitted.

        This is the mechanism behind the CIFAR-10 training failure, stated as
        a property of ``ScaleToBounds`` itself rather than of a particular
        backbone. ``_get_initial_lengthscale`` fits the lengthscale to the mean
        pairwise distance of the *unscaled* features; ``ScaleToBounds`` then
        changes that distance, so the GP trains at a scale its kernel was not
        initialized for.

        On real CIFAR-10 WRN features the effect was an expansion from 4.26 to
        16.55 against a fitted lengthscale of 4.62 -- roughly 4x, deep into RBF
        saturation, where the backbone gradient was ~81x weaker. The direction
        depends on the backbone's output scale (a small-output backbone gets
        expanded *toward* the lengthscale instead), so what is asserted here is
        the distortion, not its sign.
        """
        from gpytorch.utils.grid import ScaleToBounds

        from lightning_uq_box.uq_methods.deep_kernel_learning import (
            _get_initial_lengthscale,
        )

        torch.manual_seed(0)
        # Features on a scale where [-2, 2] is a contraction, as a real WRN's
        # penultimate features are.
        features = torch.randn(128, 64) * 5.0

        fitted_lengthscale = float(_get_initial_lengthscale(features).cpu())
        distance_before = float(torch.pdist(features).mean())

        scaler = ScaleToBounds(-2.0, 2.0)
        scaler.train()
        distance_after = float(torch.pdist(scaler(features)).mean())

        # The fitted lengthscale tracks the unscaled distances...
        assert abs(distance_before - fitted_lengthscale) / fitted_lengthscale < 0.5
        # ...and scaling moves the distances well away from it.
        assert abs(distance_after - fitted_lengthscale) / fitted_lengthscale > 0.5
