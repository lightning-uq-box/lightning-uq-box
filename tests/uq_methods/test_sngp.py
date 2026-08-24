# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""SNGP regression tests for behavior not covered by the generic config-driven
smoke tests in test_regression.py/test_classification.py.

Both bugs covered here were invisible to those smoke tests because they only
train for a single epoch, which trivially satisfies `current_epoch % 2 == 0`
and never distinguishes a buggy `feature_scale` default from `None`.
"""

import math
from pathlib import Path
from unittest.mock import patch

import torch
from conftest import minimal_trainer_kwargs
from lightning import Trainer
from torch import nn

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.uq_methods import SNGPRegression
from lightning_uq_box.uq_methods.sngp import RandomFourierFeatures


def _toy_sngp_model(**kwargs) -> SNGPRegression:
    feature_extractor = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
    return SNGPRegression(
        feature_extractor=feature_extractor,
        loss_fn=nn.MSELoss(),
        num_gp_features=8,
        num_random_features=16,
        **kwargs,
    )


class TestRandomFourierFeaturesScale:
    def test_default_feature_scale_matches_reference(self) -> None:
        """Default feature_scale must be sqrt(num_random_features / 2).

        This is the scaling used by the reference implementations (DUE,
        edward2's RandomFeatureGaussianProcess) so that the random features
        approximate the RBF kernel.
        """
        rff = RandomFourierFeatures(in_dim=8, num_random_features=1024)
        expected = math.sqrt(1024 / 2)
        assert torch.allclose(rff.feature_scale, torch.tensor(expected))

    def test_sngp_default_feature_scale_matches_rff_default(self) -> None:
        """SNGPRegression must not override RandomFourierFeatures' own default.

        Regression test: SNGPRegression/SNGPClassification used to default
        `feature_scale=2`, which silently overrode the correct
        `sqrt(num_random_features / 2)` scaling for anyone not passing
        `feature_scale=None` explicitly (an ~11x error at the library default
        of 1024 random features).
        """
        model = _toy_sngp_model()
        expected = math.sqrt(model.num_random_features / 2)
        assert torch.allclose(model.rff.feature_scale, torch.tensor(expected))


class TestSNGPCovarianceRecompute:
    def test_recomputed_every_validation_epoch(
        self, tmp_path: Path, accelerator_config: dict
    ) -> None:
        """The covariance matrix must be recomputed after every epoch.

        Regression test: `on_validation_epoch_end` used to only call
        `recompute_covariance_matrix` when `current_epoch % 2 == 0`, leaving
        predictive uncertainty stale by up to two epochs, and never refreshed
        against the final trained weights whenever training ran for an odd
        number of epochs.
        """
        model = _toy_sngp_model()
        # n_points must be large enough that the validation split (after the
        # further val/calib split) has at least two samples for R2Score.
        datamodule = ToyHeteroscedasticDatamodule(n_points=100, batch_size=8)

        n_epochs = 3
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, max_epochs=n_epochs)
        )

        with patch.object(
            model,
            "recompute_covariance_matrix",
            wraps=model.recompute_covariance_matrix,
        ) as recompute_spy:
            trainer.fit(model, datamodule=datamodule)

        assert recompute_spy.call_count == n_epochs
