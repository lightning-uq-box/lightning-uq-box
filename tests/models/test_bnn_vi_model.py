# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Test BNN VI Model Functionality."""

import pytest
import torch
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from omegaconf import OmegaConf

model_config_paths = ["tests/configs/image_regression/bnn_vi.yaml"]


class Test_BNN_VI_Model:
    # @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.fixture(params=model_config_paths)
    def bnn_model(self, request: SubRequest) -> None:
        model_config_path = request.param
        model_conf = OmegaConf.load(model_config_path)
        model = instantiate(model_conf.uq_method)

        return model

    def test_freeze_layers(self, bnn_model):
        # Freeze layers and check
        bnn_model.freeze_layers()
        for name, module in bnn_model.named_modules():
            if "Variational" in module.__class__.__name__:
                assert module.is_frozen

        # pass input through model 2 times and check if the output is same
        input = torch.randn(1, 3, 64, 64)
        output1 = bnn_model(input)
        output2 = bnn_model(input)
        assert torch.allclose(output1, output2)

    def test_unfreeze_layers(self, bnn_model):
        # Unfreeze layers and check
        bnn_model.unfreeze_layers()
        for name, module in bnn_model.named_modules():
            if "Variational" in module.__class__.__name__:
                assert not module.is_frozen

        input = torch.randn(1, 3, 64, 64)
        output1 = bnn_model(input)
        output2 = bnn_model(input)
        assert not torch.allclose(output1, output2)
