# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Tests for burnin_epochs handling in MVE models.

Regression tests for https://github.com/lightning-uq-box/lightning-uq-box/issues/150:
``MVEBase`` accepted ``burnin_epochs`` without storing it as an attribute and
``MVEPxRegression`` did not support burnin epochs at all.
"""

from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch import Tensor, nn

from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import MVEBase, MVEPxRegression, MVERegression
from lightning_uq_box.uq_methods.loss_functions import NLL


def regression_model(burnin_epochs: int) -> MVERegression:
    """Create a small MVE regression model."""
    return MVERegression(
        model=MLP(n_inputs=2, n_outputs=2, n_hidden=[50]), burnin_epochs=burnin_epochs
    )


def regression_batch() -> dict[str, Tensor]:
    """Create a small regression batch."""
    torch.manual_seed(0)
    return {"input": torch.randn(4, 2), "target": torch.randn(4, 1)}


def px_model(burnin_epochs: int = 0) -> MVEPxRegression:
    """Create a small MVE pixelwise regression model."""
    return MVEPxRegression(
        model=nn.Conv2d(3, 2, kernel_size=1), burnin_epochs=burnin_epochs
    )


def px_batch() -> dict[str, Tensor]:
    """Create a small pixelwise regression batch."""
    torch.manual_seed(0)
    return {"input": torch.randn(2, 3, 8, 8), "target": torch.randn(2, 1, 8, 8)}


class TestMVEBurnin:
    """Burnin behaviour of MVEBase and MVERegression."""

    def test_base_stores_burnin_epochs(self) -> None:
        """MVEBase stores burnin_epochs as an instance attribute."""
        model = MVEBase(
            model=MLP(n_inputs=2, n_outputs=2, n_hidden=[50]), burnin_epochs=3
        )
        assert model.burnin_epochs == 3

    def test_base_training_step_during_burnin_uses_mse(self) -> None:
        """Bare MVEBase computes the burnin loss from the stored attribute.

        Unlike MVERegression, the bare base never calls
        ``save_hyperparameters``, so its ``hparams`` hold no burnin_epochs:
        this fails if training_step reads ``self.hparams.burnin_epochs``
        instead of the attribute.
        """
        model = MVEBase(
            model=MLP(n_inputs=2, n_outputs=2, n_hidden=[50]), burnin_epochs=5
        )
        batch = regression_batch()
        with patch.object(model, "log"):
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        expected = nn.functional.mse_loss(out[:, 0:1], batch["target"])
        torch.testing.assert_close(loss, expected)

    def test_regression_stores_burnin_epochs(self) -> None:
        """MVERegression exposes the burnin_epochs it was constructed with."""
        assert regression_model(burnin_epochs=3).burnin_epochs == 3

    def test_training_step_during_burnin_uses_mse(self) -> None:
        """Before the burnin epochs elapse, the loss is MSE on the mean."""
        model = regression_model(burnin_epochs=5)
        batch = regression_batch()
        with patch.object(model, "log"):
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        expected = nn.functional.mse_loss(out[:, 0:1], batch["target"])
        torch.testing.assert_close(loss, expected)

    def test_training_step_after_burnin_uses_nll(self) -> None:
        """With no burnin epochs, the loss is the NLL from the first epoch."""
        model = regression_model(burnin_epochs=0)
        batch = regression_batch()
        with patch.object(model, "log"):
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        expected = NLL()(out, batch["target"])
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("epoch,uses_mse", [(1, True), (2, False)])
    def test_training_step_switches_at_burnin_epoch(
        self, epoch: int, uses_mse: bool
    ) -> None:
        """The loss switches from MSE to NLL exactly at burnin_epochs."""
        model = regression_model(burnin_epochs=2)
        batch = regression_batch()
        with (
            patch.object(model, "log"),
            patch.object(
                MVERegression, "current_epoch", new_callable=PropertyMock
            ) as mock_epoch,
        ):
            mock_epoch.return_value = epoch
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        if uses_mse:
            expected = nn.functional.mse_loss(out[:, 0:1], batch["target"])
        else:
            expected = NLL()(out, batch["target"])
        torch.testing.assert_close(loss, expected)


class TestMVEPxBurnin:
    """Burnin behaviour of MVEPxRegression."""

    def test_default_burnin_epochs_is_zero(self) -> None:
        """The default keeps the previous behaviour: NLL from epoch zero."""
        model = MVEPxRegression(model=nn.Conv2d(3, 2, kernel_size=1))
        assert model.burnin_epochs == 0

    def test_stores_burnin_epochs(self) -> None:
        """MVEPxRegression stores burnin_epochs as an instance attribute."""
        assert px_model(burnin_epochs=3).burnin_epochs == 3

    def test_positional_arguments_keep_their_meaning(self) -> None:
        """Positional args bind as they did before burnin_epochs existed."""
        model = MVEPxRegression(nn.Conv2d(3, 2, kernel_size=1), True, True)
        assert model.freeze_backbone
        assert model.freeze_decoder
        assert model.burnin_epochs == 0

    def test_training_step_during_burnin_uses_mse(self) -> None:
        """Before the burnin epochs elapse, the loss is MSE on the mean."""
        model = px_model(burnin_epochs=5)
        batch = px_batch()
        with patch.object(model, "log"):
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        expected = nn.functional.mse_loss(out[:, 0:1, ...], batch["target"])
        torch.testing.assert_close(loss, expected)

    def test_training_step_after_burnin_uses_nll(self) -> None:
        """With no burnin epochs, the loss is the NLL from the first epoch."""
        model = px_model(burnin_epochs=0)
        batch = px_batch()
        with patch.object(model, "log"):
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        expected = NLL()(out, batch["target"])
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("epoch,uses_mse", [(1, True), (2, False)])
    def test_training_step_switches_at_burnin_epoch(
        self, epoch: int, uses_mse: bool
    ) -> None:
        """The loss switches from MSE to NLL exactly at burnin_epochs."""
        model = px_model(burnin_epochs=2)
        batch = px_batch()
        with (
            patch.object(model, "log"),
            patch.object(
                MVEPxRegression, "current_epoch", new_callable=PropertyMock
            ) as mock_epoch,
        ):
            mock_epoch.return_value = epoch
            loss = model.training_step(batch, 0)
        out = model.forward(batch["input"])
        if uses_mse:
            expected = nn.functional.mse_loss(out[:, 0:1, ...], batch["target"])
        else:
            expected = NLL()(out, batch["target"])
        torch.testing.assert_close(loss, expected)
