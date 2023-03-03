#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
from vc_models.transforms.to_tensor_if_not import ToTensorIfNot


class RandomizeEnvTransform:
    def __init__(self, transform, randomize_environments=False):
        self.apply = transform
        self.randomize_environments = randomize_environments

    def __call__(
        self,
        x: torch.Tensor,
        N: Optional[int] = None,
    ):
        x = ToTensorIfNot()(x)
        single_img = x.ndim == 3
        if single_img:
            x = x.unsqueeze(0)

        if not self.randomize_environments or N is None:
            x = self.apply(x)
        else:
            # shapes
            TN = x.size(0)
            T = TN // N

            # apply the same augmentation when t == 1 for speed
            # typically, t == 1 during policy rollout
            if T == 1:
                x = self.apply(x)
            else:
                # put environment (n) first
                _, A, B, C = x.shape
                x = torch.einsum("tnabc->ntabc", x.view(T, N, A, B, C))

                # apply the same transform within each environment
                x = torch.cat([self.apply(imgs) for imgs in x])

                # put timestep (t) first
                _, A, B, C = x.shape
                x = torch.einsum("ntabc->tnabc", x.view(N, T, A, B, C)).flatten(0, 1)

        if single_img:
            return x[0]
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.apply})"
