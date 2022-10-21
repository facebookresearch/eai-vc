#!/usr/bin/env python3

from typing import Callable, Union

import torch
from omnivore.models.vision_transformer import Attention, Mlp, trunc_normal_

from timm.models.vision_transformer import DropPath  # noqa: F401
from torch import nn, Tensor


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class BlockWithEfficientDropPath(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        layer_scale_type: str = None,
        layer_scale_init_value: float = 1e-4,
        ffn_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_target()

        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, attn_mask: torch.Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            if self.layer_scale_type is None:
                return self.attn(self.norm1(x), attn_mask)
            else:
                return self.layer_scale_gamma1 * (self.attn(self.norm1(x), attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            if self.layer_scale_type is None:
                return self.mlp(self.norm2(x))
            else:
                return self.layer_scale_gamma2 * (self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.0:
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)
