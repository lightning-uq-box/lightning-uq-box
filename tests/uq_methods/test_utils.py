"""Test UQ-methods utils."""

import pytest
from torch.nn import MSELoss

from uq_method_box.train_utils.loss_functions import NLL, QuantileLoss
from uq_method_box.uq_methods.utils import retrieve_loss_fn


@pytest.mark.parametrize(
    "loss_fn_name,expected_class",
    [("nll", NLL), ("quantile", QuantileLoss), ("mse", MSELoss)],
)
def test_retrieve_loss_fn(loss_fn_name, expected_class):
    loss_fn = retrieve_loss_fn(loss_fn_name)
    assert isinstance(loss_fn, expected_class)


def test_invalid_loss_fn(loss_fn_name="foo"):
    with pytest.raises(ValueError, match="Your loss function choice is not supported"):
        retrieve_loss_fn(loss_fn_name)
