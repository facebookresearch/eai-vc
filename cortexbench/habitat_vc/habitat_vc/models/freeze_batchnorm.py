#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch


def convert_frozen_batchnorm(module):
    r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
    :class:`torch.nn.FrozenBatchNorm` layers.
    Args:
        module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
        process_group (optional): process group to scope synchronization,
            default is the whole world
    Returns:
        The original :attr:`module` with the converted :class:`torch.nn.FrozenBatchNorm`
        layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
        a new :class:`torch.nn.FrozenBatchNorm` layer object will be returned
        instead.
    Example::
        >>> # Network with nn.BatchNorm layer
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100),
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> # xdoctest: +SKIP("distributed")
        >>> frozen_bn_module = convert_frozen_batchnorm(module)
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = _FrozenBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_batchnorm(child))
    del module
    return module_output


class _FrozenBatchNorm(torch.nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        # if self.training:
        #     bn_training = True
        # else:
        #     bn_training = (self.running_mean is None) and (self.running_var is None)
        bn_training = False

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        return
