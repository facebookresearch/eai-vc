from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class HeadArgs:
    head: torch.nn.Module
    num_tokens: int


class NumTokenSpecificHead(torch.nn.Module):
    def __init__(self, heads: List[Dict], num_tokens_dim: int = 1) -> None:
        super().__init__()
        # validate the input
        heads = [HeadArgs(**x) for x in heads]
        num_tokens = [x.num_tokens for x in heads]
        assert len(num_tokens) == len(set(num_tokens)), "Found duplicates in num_tokens"
        self.token_to_head_idx = {num_tokens[x]: x for x in range(len(num_tokens))}
        self.heads = torch.nn.ModuleList([x.head for x in heads])
        self.num_tokens_dim = num_tokens_dim

    def forward(self, x: torch.Tensor):
        head_idx = self.token_to_head_idx[x.shape[self.num_tokens_dim]]
        return self.heads[head_idx](x)
