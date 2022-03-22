from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, RandomApply


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Transform:
    is_random: bool = False

    def apply(self, x: torch.Tensor):
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        N: Optional[int] = None,
        V: Optional[int] = None,
        skip_one_T: bool = True,
    ):
        if not self.is_random:
            return self.apply(x)

        if None in (T, N, V):
            return self.apply(x)

        if T == 1 and skip_one_T:
            return self.apply(x)

        # put environment (n) first
        _, A, B, C = x.shape
        x = torch.einsum("tnvabc->ntvabc", x.view(T, N, V, A, B, C)).flatten(1, 2)

        # apply the same transform within each environment
        x = torch.cat([self.apply(imgs) for imgs in x])

        # put timestep (t) first
        _, A, B, C = x.shape
        x = torch.einsum("ntvabc->tnvabc", x.view(N, T, V, A, B, C)).flatten(0, 2)

        return x


class ResizeTransform(Transform):
    def __init__(self, size):
        self.size = size

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.resize(x, self.size)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        return x


class ShiftAndJitterTransform(Transform):
    is_random: bool = True

    def __init__(self, size):
        self.size = size

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        x = TF.resize(x, self.size)
        x = TF.center_crop(x, output_size=self.size)
        x = x.float() / 255.0
        x = RandomShiftsAug(4)(x)
        x = RandomApply([ColorJitter(0.3, 0.3, 0.3, 0.3)], p=1.0)(x)
        return x


def get_transform(name, size):
    if name == "resize":
        return ResizeTransform(size)
    elif name == "shift+jitter":
        return ShiftAndJitterTransform(size)
