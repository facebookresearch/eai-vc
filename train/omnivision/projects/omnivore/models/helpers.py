import math

import einops
import numpy as np
import torch

import torch.nn as nn

from omnivision.utils.distributed import all_reduce_mean


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class NormalizeAndCenter(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("center", torch.zeros(1, out_features))
        self.center_momentum = center_momentum

    def forward(self, x):
        x = x - self.center
        x = torch.nn.functional.normalize(x, dim=self.dim, p=2)
        self.update_center(x)
        return x

    @torch.no_grad()
    def update_center(self, x):
        """
        Source: https://github.com/facebookresearch/dino/blob/main/main_dino.py
        """
        batch_center = torch.mean(x, dim=0, keepdim=True)
        all_reduce_mean(batch_center)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class NormalizeAndBatchNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.affine_bn = nn.SyncBatchNorm(out_features, affine=False)

    def forward(self, x):
        x = self.affine_bn(x)
        x = torch.nn.functional.normalize(x, dim=self.dim, p=2)
        return x


class LearnableLogitScaling(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable},max_logit_scale={self.max_logit_scale}"
        return st


def singleton_dict_to_item(output: dict):
    assert isinstance(output, dict) and len(output) == 1
    return output[list(output.keys())[0]]


def lecun_normal_init(tensor: torch.Tensor, fan_in: int):
    # Following classy_vision/models/lecun_normal_init/lecun_normal_init.py
    return nn.init.trunc_normal_(tensor, std=math.sqrt(1 / fan_in))


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


class EinOpsReduce(nn.Module):
    def __init__(self, reduce_expr: str, reduce_op: str, **kwargs) -> None:
        super().__init__()
        self.reduce_expr = reduce_expr
        self.reduce_op = reduce_op
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.reduce(
            x, self.reduce_expr, reduction=self.reduce_op, **self.kwargs
        )


class VerboseNNModule(nn.Module):
    """
    Wrapper around nn.Module that prints registered buffers and parameter names.
    """

    @staticmethod
    def get_readable_tensor_repr(name: str, tensor: torch.Tensor) -> str:
        st = (
            "("
            + name
            + "): "
            + "tensor("
            + str(tuple(tensor[1].shape))
            + ", requires_grad="
            + str(tensor[1].requires_grad)
            + ")\n"
        )
        return st

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr += self.get_readable_tensor_repr(name, p)

        for p in self.named_buffers():
            name = p[0].split(".")[0]
            string_repr += self.get_readable_tensor_repr(name, p)

        return string_repr


class NoGradWrapper(nn.Module):
    """
    Simple module which allows to wrap the forward with
    torch.no_grad()
    """

    def __init__(self, module: nn.Module, set_to_eval: bool = True):
        super().__init__()
        self.module = module
        self.set_to_eval = set_to_eval
        if set_to_eval:
            self.eval()

    def train(self, model: bool):
        if self.eval:
            self.module.eval()
        else:
            self.module.train()

    def eval(self):
        self.module.eval()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def cast_if_src_dtype(
    tensor: torch.Tensor, src_dtype: torch.dtype, tgt_dtype: torch.dtype
):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated
