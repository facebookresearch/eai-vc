from typing import Mapping

import torch
import torch.nn as nn


class DictApplyLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, key: str) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.key = key

    def forward(self, input: Mapping, *args, **kwargs):
        assert isinstance(input, Mapping)
        return self.loss_fn(input[self.key], *args, **kwargs)
