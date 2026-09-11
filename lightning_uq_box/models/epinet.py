# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

r"""Network modules for Epistemic Neural Networks (epinet).

Implements the building blocks of the epinet introduced in

* https://arxiv.org/abs/2107.08924

An epinet augments a conventional "base" network :math:`\mu_\zeta(x)` with a small
head :math:`\sigma_\eta` that additionally consumes an *epistemic index*
:math:`z \sim \mathcal{N}(0, I_{D_Z})`, so that the combined network produces useful
*joint* predictions over several inputs rather than only marginals.
"""

import torch
from torch import Tensor, nn


class ProjectedMLP(nn.Module):
    r"""MLP whose output is projected onto the epistemic index.

    Implements :math:`\sigma^L_\eta(\tilde{x}, z) = \mathrm{mlp}_\eta([\tilde{x},
    z])^T z`, equation 6 of https://arxiv.org/abs/2107.08924. The MLP produces
    ``n_outputs * index_dim`` values that are reshaped to
    ``[batch_size, n_outputs, index_dim]`` and contracted with the index.

    Unlike the reference JAX implementation, which asserts a single index shared by the
    whole batch, this module accepts a *per-example* index of shape
    ``[batch_size, index_dim]``. A shared index is recovered with ``z.expand(B, -1)``.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        n_outputs: int,
        index_dim: int,
        concat_index: bool = True,
    ) -> None:
        r"""Initialize a new instance of ProjectedMLP.

        Args:
            in_features: number of input features :math:`\tilde{x}`
            hidden_dims: sizes of the hidden layers
            n_outputs: number of outputs of the combined network
            index_dim: dimension :math:`D_Z` of the epistemic index
            concat_index: whether to concatenate the index to the input features
        """
        super().__init__()

        self.in_features = in_features
        self.n_outputs = n_outputs
        self.index_dim = index_dim
        self.concat_index = concat_index

        mlp_in = in_features + index_dim if concat_index else in_features
        layer_sizes = [mlp_in, *hidden_dims]
        layers: list[nn.Module] = []
        for idx in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-1], n_outputs * index_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor, index: Tensor) -> Tensor:
        """Forward pass.

        Args:
            features: input features of shape [batch_size, in_features]
            index: epistemic index of shape [batch_size, index_dim]

        Returns:
            output of shape [batch_size, n_outputs]
        """
        inputs = torch.cat([features, index], dim=-1) if self.concat_index else features
        out = self.mlp(inputs)
        # [batch_size, n_outputs, index_dim], so that the *last* axis is the one
        # contracted with the index. A transposed reshape here still runs and still
        # trains, but implements a different function, so it is pinned by a test.
        out = out.reshape(-1, self.n_outputs, self.index_dim)
        return torch.einsum("boi,bi->bo", out, index)


class EnsemblePriorFunction(nn.Module):
    r"""Frozen ensemble of MLPs over the raw input, combined linearly in the index.

    Represents the additive prior :math:`\sigma^P` over the raw network input. The
    ensemble members are independently initialized and *never trained*: all of their
    parameters have ``requires_grad=False``, and because they are ordinary submodules
    they move with ``.to(device)`` and are written to and restored from checkpoints
    rather than being re-randomized on reload.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        num_ensemble: int,
        hidden_dims: list[int],
        seed: int = 0,
    ) -> None:
        """Initialize a new instance of EnsemblePriorFunction.

        Args:
            n_inputs: number of input features, the flattened input dimension
            n_outputs: number of outputs of each ensemble member
            num_ensemble: number of ensemble members, usually the index dimension
            hidden_dims: sizes of the hidden layers of each member
            seed: seed used to initialize the ensemble members
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.num_ensemble = num_ensemble

        generator = torch.Generator().manual_seed(seed)
        members: list[nn.Module] = []
        for _ in range(num_ensemble):
            layer_sizes = [n_inputs, *hidden_dims]
            layers: list[nn.Module] = []
            for idx in range(1, len(layer_sizes)):
                layers += [nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]), nn.ReLU()]
            layers += [nn.Linear(layer_sizes[-1], n_outputs)]
            member = nn.Sequential(*layers)
            _reinit_with_generator(member, generator)
            members.append(member)

        self.members = nn.ModuleList(members)
        self.requires_grad_(False)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x: raw input of shape [batch_size, \*input_shape]
            index: epistemic index of shape [batch_size, index_dim]

        Returns:
            output of shape [batch_size, n_outputs]
        """
        flat = torch.flatten(x, 1)
        outs = torch.stack([member(flat) for member in self.members], dim=0)
        return torch.einsum("nbo,bn->bo", outs, index)


class ConvEnsemblePriorFunction(nn.Module):
    """Frozen convolutional ensemble prior for image inputs.

    The image counterpart of :class:`EnsemblePriorFunction`, matching the CIFAR-10
    prior of the reference implementation: three ``5 x 5`` stride-2 convolutions with
    output channels ``(4, 8, 4)``, ReLU in between, flattened into a linear read-out.
    As with the MLP ensemble every member is frozen and checkpointed.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        num_ensemble: int,
        input_size: int = 32,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        stride: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialize a new instance of ConvEnsemblePriorFunction.

        Args:
            in_channels: number of channels of the input image
            n_outputs: number of outputs of each ensemble member
            num_ensemble: number of ensemble members, usually the index dimension
            input_size: spatial size of the (square) input image
            channels: output channels of the convolutions, defaults to ``[4, 8, 4]``
            kernel_size: convolution kernel size
            stride: convolution stride
            seed: seed used to initialize the ensemble members
        """
        super().__init__()

        if channels is None:
            channels = [4, 8, 4]

        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.num_ensemble = num_ensemble

        # spatial size after the convolution stack, with the default
        # `padding=0` of nn.Conv2d
        size = input_size
        for _ in channels:
            size = (size - kernel_size) // stride + 1
        if size < 1:
            raise ValueError(
                f"Input size {input_size} is too small for {len(channels)} "
                f"convolutions with kernel_size={kernel_size} and stride={stride}."
            )
        flat_dim = channels[-1] * size * size

        generator = torch.Generator().manual_seed(seed)
        members: list[nn.Module] = []
        for _ in range(num_ensemble):
            layers: list[nn.Module] = []
            prev = in_channels
            for out_channels in channels:
                layers += [
                    nn.Conv2d(prev, out_channels, kernel_size, stride=stride),
                    nn.ReLU(),
                ]
                prev = out_channels
            layers += [nn.Flatten(), nn.Linear(flat_dim, n_outputs)]
            member = nn.Sequential(*layers)
            _reinit_with_generator(member, generator)
            members.append(member)

        self.members = nn.ModuleList(members)
        self.requires_grad_(False)

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input images of shape [batch_size, in_channels, height, width]
            index: epistemic index of shape [batch_size, index_dim]

        Returns:
            output of shape [batch_size, n_outputs]
        """
        outs = torch.stack([member(x) for member in self.members], dim=0)
        return torch.einsum("nbo,bn->bo", outs, index)


class Epinet(nn.Module):
    r"""Epinet head, a learnable projected MLP plus frozen prior functions.

    Implements :math:`\sigma_\eta(\tilde{x}, z) = \sigma^L_\eta(\tilde{x}, z) +
    \sigma^P(\tilde{x}, z)`, equation 5 of https://arxiv.org/abs/2107.08924. Two
    prior paths exist and both scales are configurable: the epinet's own frozen
    :class:`ProjectedMLP` prior over the features, and an additive
    :class:`EnsemblePriorFunction` prior over the raw input.

    The defaults follow the Neural Testbed agent, where the epinet's own prior is
    scaled to zero and all prior variation comes from the additive ensemble prior over
    the raw input. Image experiments use the mirror image of this, an
    ``epi_prior_scale`` of 4.0 with ``input_prior_scale=0.0`` and a convolutional
    prior passed as ``input_prior``.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        n_feature_inputs: int,
        n_raw_inputs: int,
        n_outputs: int,
        index_dim: int = 8,
        hidden_dims: list[int] | None = None,
        prior_hidden_dims: list[int] | None = None,
        epi_prior_scale: float = 0.0,
        input_prior_scale: float = 0.3,
        seed: int = 0,
        input_prior: nn.Module | None = None,
    ) -> None:
        r"""Initialize a new instance of Epinet.

        Args:
            n_feature_inputs: dimension of the features :math:`\phi(x)` handed to the
                epinet
            n_raw_inputs: flattened dimension of the raw network input, used by the
                additive prior over the input
            n_outputs: number of outputs of the base network
            index_dim: dimension :math:`D_Z` of the epistemic index
            hidden_dims: hidden sizes of the learnable epinet, defaults to ``[15, 15]``
            prior_hidden_dims: hidden sizes of the prior networks, defaults to
                ``[5, 5]``
            epi_prior_scale: scale of the epinet's own frozen prior over the features
            input_prior_scale: scale of the additive prior over the raw input, which is
                skipped entirely when this is ``0.0``
            seed: seed used to initialize the frozen prior networks
            input_prior: optional prior module over the raw input, for instance a
                :class:`ConvEnsemblePriorFunction` for image inputs. When ``None`` an
                :class:`EnsemblePriorFunction` is built from ``prior_hidden_dims``.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [15, 15]
        if prior_hidden_dims is None:
            prior_hidden_dims = [5, 5]

        self.index_dim = index_dim
        self.n_outputs = n_outputs
        self.epi_prior_scale = epi_prior_scale
        self.input_prior_scale = input_prior_scale

        self.train_epinet = ProjectedMLP(
            n_feature_inputs, hidden_dims, n_outputs, index_dim
        )

        # the epinet's own prior: same shape, frozen, separately initialized
        epi_prior = ProjectedMLP(n_feature_inputs, hidden_dims, n_outputs, index_dim)
        _reinit_with_generator(epi_prior, torch.Generator().manual_seed(seed + 1))
        epi_prior.requires_grad_(False)
        self.epi_prior = epi_prior

        self.input_prior: nn.Module | None
        if input_prior is not None:
            input_prior.requires_grad_(False)
            self.input_prior = input_prior
        elif input_prior_scale != 0.0:
            self.input_prior = EnsemblePriorFunction(
                n_raw_inputs, n_outputs, index_dim, prior_hidden_dims, seed=seed
            )
        else:
            self.input_prior = None

    def forward(self, features: Tensor, x: Tensor, index: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            features: features :math:`\phi(x)` of shape
                [batch_size, n_feature_inputs], expected to be stop-gradiented by the
                caller
            x: raw network input of shape [batch_size, \*input_shape]
            index: epistemic index of shape [batch_size, index_dim]

        Returns:
            the epinet contribution of shape [batch_size, n_outputs]
        """
        out = self.train_epinet(features, index)
        if self.epi_prior_scale != 0.0:
            out = out + self.epi_prior_scale * self.epi_prior(features, index)
        if self.input_prior is not None:
            out = out + self.input_prior_scale * self.input_prior(x, index)
        return out


def _reinit_with_generator(module: nn.Module, generator: torch.Generator) -> None:
    """Re-initialize the parameters of a module from a seeded generator.

    Gives every ensemble member independent, reproducible weights without disturbing
    the global RNG state, which callers may be relying on for their own seeding.

    Args:
        module: module whose ``Linear``/``Conv2d`` parameters are re-initialized
        generator: seeded random number generator advanced by this call
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear | nn.Conv2d):
            fan_in = layer.weight[0].numel()
            bound = 1.0 / (fan_in**0.5)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                if layer.bias is not None:
                    layer.bias.uniform_(-bound, bound, generator=generator)
