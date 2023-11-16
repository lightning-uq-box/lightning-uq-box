"""Spectral Normalization Layers and conversion tools."""

"""Adapted from https://github.com/y0ast/DUE/tree/main/due/layers"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import conv2d, conv_transpose2d, normalize
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)


def spectral_normalize_model_layers(
    model: nn.Module,
    n_power_iterations: int,
    input_dimensions: dict[str, torch.Size],
    coeff=0.95,
) -> nn.Module:
    """Convert layers of a standard model into spectral normalized layers.

    Effectively replace normal layers with spectral normalized layer version.

    Args:
        model: model to spectral normalize layers for
        n_power_iterations: number of power iterations in spectral norm layers
        input_dimensions: dictionary holding layer module name and input dimension to that
            layer, which is necessary for spectral normalized conv layers
        coeff: soft normalization only when sigma larger than coeff
    """
    for name, _ in list(model._modules.items()):
        if model._modules[name]._modules:
            spectral_normalize_model_layers(
                model._modules[name], n_power_iterations, input_dimensions, coeff
            )
        elif "Linear" in model._modules[name].__class__.__name__:
            setattr(
                model,
                name,
                spectral_norm_fc(
                    model._modules[name],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                ),
            )
        elif "Conv2d" in model._modules[name].__class__.__name__:
            # TODO: need to get input dimension (3,32,32) for example to this conv layer
            setattr(
                model,
                name,
                spectral_norm_conv(
                    model._modules[name],
                    coeff=coeff,
                    input_dim=input_dimensions[str(id(model._modules[name]))],
                    n_power_iterations=n_power_iterations,
                ),
            )
        else:
            pass

    return model


class _SpectralBatchNorm(_NormBase):
    def __init__(
        self, num_features, coeff, eps=1e-5, momentum=0.01, affine=True
    ):  # momentum is 0.01 by default instead of 0.1 of BN which alleviates
        # noisy power iteration
        # Code is based on torch.nn.modules._NormBase
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)
        self.coeff = coeff

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement here to tell the jit to skip emitting when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization
        rather than the buffers. Mini-batch stats are used in training mode, and
        in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers only updated if they are to be tracked and we are in training mode.
        Thus they only need to be passed when the update should occur
        (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # before the foward pass, estimate the lipschitz constant of the layer and
        # divide by it so that the lipschitz constant of the batch norm operator is
        # approximately 1
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )
        # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
        lipschitz = torch.max(torch.abs(weight * (self.running_var + self.eps) ** -0.5))

        # if lipschitz of the operation is greater than coeff, then we want to divide
        # the input by constant to force the overall lipchitz factor of the batch norm
        # to be exactly coeff
        lipschitz_factor = torch.max(lipschitz / self.coeff, torch.ones_like(lipschitz))

        weight = weight / lipschitz_factor

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SpectralBatchNorm1d(_SpectralBatchNorm, nn.BatchNorm1d):
    """Spectral Normalized Batch Norm 1D."""

    pass


class SpectralBatchNorm2d(_SpectralBatchNorm, nn.BatchNorm2d):
    """Spectral Normalized Batch Norm 2D."""

    pass


class SpectralBatchNorm3d(_SpectralBatchNorm, nn.BatchNorm3d):
    """Spectral Normalized Batch Norm 3D."""

    pass


"""
From: https://github.com/jhjacobsen/invertible-resnet
Which is based on: https://arxiv.org/abs/1811.00995

Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""


class SpectralNormConv(SpectralNorm):
    """Spectral Norm Convolutional Layer."""

    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        """Compute spectral normalized weight.

        Args:
            module:
            do_power_iteration: whether or not to apply power iterations

        Returns:
            computed weight tensor
        """
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # get settings from conv-module (for transposed convolution parameters)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                output_padding = 0
                if stride[0] > 1:
                    # Note: the below does not generalize to stride > 2
                    output_padding = 1 - self.input_dim[-1] % 2

                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.output_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(
                        v.view(self.input_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        bias=None,
                    )
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        weight_v = conv2d(
            v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    def __call__(self, module, inputs) -> None:
        """Call module.

        Args:
            module:
            inputs:
        """
        assert (
            inputs[0].shape[1:] == self.input_dim[1:]
        ), "Input dims don't match actual input"
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        input_dim: Tuple[int],
        name: str,
        n_power_iterations: int,
        eps: float,
    ) -> "SpectralNormConv":
        """Apply spectral normalization to Conv layer.

        Args:
            module:
            coeff: soft normalization only when sigma larger than coeff
            input_dim:
            name:
            n_power_iterations:
            eps:

        """
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormConv(name, n_power_iterations, eps=eps)
        fn.coeff = coeff
        fn.input_dim = input_dim
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(
                v.view(input_dim), weight, stride=stride, padding=padding, bias=None
            )
            fn.output_dim = u.shape
            num_output_dim = (
                fn.output_dim[0]
                * fn.output_dim[1]
                * fn.output_dim[2]
                * fn.output_dim[3]
            )
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_conv(
    module, coeff, input_dim: torch.Size, n_power_iterations=1, name="weight", eps=1e-12
):
    """Apply spectral normalization to Convolutions with flexible max norm.

    Args:
        module (nn.Module): containing convolution module
        input_dim (tuple(int, int, int)): dimension of input to convolution
        coeff (float, optional): coefficient to normalize to, soft normalization only when sigma larger than coeff
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_conv(nn.Conv2D(3, 16, 3), (3, 32, 32), 2.0)

    """
    input_dim_4d = torch.Size([1, input_dim[0], input_dim[1], input_dim[2]])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)

    return module


"""
Spectral Normalization from https://arxiv.org/abs/1802.05957

with additional variable `coeff` or max spectral norm.
"""


class SpectralNormFC(SpectralNorm):
    """Spectral Norm Fully Connected."""

    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        """Compute spectral normalized weight.

        Args:
            module:
            do_power_iteration

        Returns:
            computed weight tensor
        """
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNormFC":
        """Apply spectral normalization.

        Args:
            module:
            coeff:
            name:
            n_power_iterations:
            dim:
            eps:

        Returns:
            spectral normalized layer
        """
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormFC(name, n_power_iterations, dim, eps)
        fn.coeff = coeff

        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_fc(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: int = None,
):
    """Apply spectral normalization.

    Args:
        module (nn.Module): containing module
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_fc(nn.Linear(20, 40), 2.0)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNormFC.apply(module, coeff, name, n_power_iterations, dim, eps)
    return module
