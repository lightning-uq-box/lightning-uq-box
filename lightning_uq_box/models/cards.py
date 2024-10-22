# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""CARDS Model Utilities."""

import torch
import torch.nn as nn
from torch import Tensor

from lightning_uq_box.uq_methods.utils import _get_output_layer_name_and_module


class ConditionalLinear(nn.Module):
    """Conditional Linear Layer."""

    def __init__(self, n_inputs: int, n_outputs: int, n_steps: int) -> None:
        """Initialize a new instance of the layer.

        Args:
            n_inputs: number of inputs to the layer
            n_outputs: number of outputs from the layer
            n_steps: number of diffusion steps in embedding

        """
        super().__init__()
        self.n_outputs = n_outputs
        self.lin = nn.Linear(n_inputs, n_outputs)
        self.embed = nn.Embedding(n_steps, n_outputs)
        self.embed.weight.data.uniform_()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass of conditional linear layer.

        Args:
            x: input of shape [N, n_inputs]
            t: input of shape [1]

        Returns:
            output from condtitional linear model of shape [N, n_outputs]
        """
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.n_outputs) * out
        return out


class DiffusionSequential(nn.Sequential):
    """My Sequential to accept multiple inputs."""

    def forward(self, input: Tensor, t: Tensor):
        """Forward pass.

        Args:
            input: input tensor to model shape [n, feature_dim]
            t: time steps shape [1]

        Returns:
            output of diffusion model [n, output_dim]
        """
        for module in self._modules.values():
            if isinstance(module, ConditionalLinear):
                input = module(input, t)
            else:
                input = module(input)
        return input


class ConditionalGuidedLinearModel(nn.Module):
    """Conditional Guided Model."""

    def __init__(
        self,
        n_steps: int,
        x_dim: int,
        y_dim: int,
        n_hidden: list[int] = [64, 64],
        cat_x: bool = False,
        cat_y_pred: bool = False,
        activation_fn: nn.Module | None = None,
    ) -> None:
        """Initialize a new instance of Conditional Guided Model.

        Args:
            n_steps: number of diffusion steps
            x_dim: feature dimension of the x input data
            y_dim: output dimension of conditional mean model
            n_hidden: number of Conditional Linear Layers with dimension
            cat_x: whether to condition on the input x throught concatenation
                p_sample_loop would pass x to each diffusion step through concatenation
                and that improves sample quality
            cat_y_pred: whether to condition on the y_0_hat prediction
                of the conditional mean model by concatenation
            activation_fn: activation function between conditional linear layers
        """
        super().__init__()

        if activation_fn is None:
            activation_fn = nn.Softplus()
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        layer_sizes = [data_dim] + n_hidden
        layers = []
        for idx in range(1, len(layer_sizes)):
            layers += [
                ConditionalLinear(layer_sizes[idx - 1], layer_sizes[idx], n_steps),
                activation_fn,
            ]
        # final output layer is standard layer
        # layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        layers += [nn.Linear(layer_sizes[-1], y_dim)]
        self.model = DiffusionSequential(*layers)

    def forward(self, x: Tensor, y_t: Tensor, y_0_hat: Tensor, t: Tensor) -> Tensor:
        """Forward pass of the Conditional Guided Model.

        Args:
            x: input data
            y_t: target data
            y_0_hat: y_0_hat
            t: time step
        """
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        return self.model(eps_pred, t)


class ConditionalGuidedConvModel(nn.Module):
    """Conditional Guidance Model for Image tasks."""

    def __init__(
        self, encoder: nn.Module, cond_guide_model: ConditionalGuidedLinearModel
    ) -> None:
        """Initialize a new instance of Conditional Guided Conv Model.

        Args:
            encoder: encoder model acting like a feature extractor before
                a conditional linear guidance model
            cond_guide_model: conditional
            n_steps: number of diffusion steps

        Raises:
            Assertionerror for misconfigurations between encoder
                and cond_guide_model
        """
        super().__init__()

        # TODO assertion checks between the configs of the encoder and cond guidance
        # model
        # TODO assert that cat_x and cat_y_pred are false, but maybe you can as well?
        # no I think cat_x has to be false because cannot input the image and y_0_hat
        # would be the feature extraction
        assert cond_guide_model.cat_x is False, "Cannot concatenate x"
        # assert cond_guide_model.cat_y_pred is False, "Cannot concatenate y"

        self.encoder = encoder
        self.cond_guide_model = cond_guide_model
        self.n_steps = cond_guide_model.n_steps

        _, module = _get_output_layer_name_and_module(self.encoder)
        encoder_out_features = module.out_features

        assert (
            encoder_out_features == cond_guide_model.x_dim
        ), "Encoder output features has to match the x_dim of the guide model"
        self.norm = nn.BatchNorm1d(encoder_out_features)

        # "connection" modules
        self.connect_module = DiffusionSequential(
            ConditionalLinear(encoder_out_features, encoder_out_features, self.n_steps),
            nn.BatchNorm1d(encoder_out_features, encoder_out_features),
            nn.Softplus(),
        )

    def forward(self, x: Tensor, y_t: Tensor, y_0_hat: Tensor, t: Tensor) -> Tensor:
        """Forward pass of the Conditional Guided Conv Model.

        Args:
            x: input data
            y_t: target data
            y_0_hat: y_0_hat
            t: time step

        Returns:
            output of the conditional guided convolutional model
        """
        # encoding
        x = self.encoder(x)
        x = self.norm(x)

        x = self.connect_module(x, t)

        # TODO not sure about this concatentation
        # y = torch.cat([y_t, y_0_hat], dim=-1)
        # y = x * y
        # y = x * y_t

        return self.cond_guide_model(x=None, y_t=y_t, y_0_hat=y_0_hat, t=t)
