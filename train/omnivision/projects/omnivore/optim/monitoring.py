import json
import os

import torch.nn as nn
from iopath.common.file_io import g_pathmgr


class GradientNormWatcher:
    """
    Utility tool to watch the norm of the parameters and their gradients
    and log them
    """

    def __init__(self, output_path: str, accum_steps: int = 1, detailed: bool = False):
        super(GradientNormWatcher, self).__init__()
        self.output_path = output_path
        self.accum_steps = accum_steps
        self.current_step = 0
        self.detailed = detailed
        self.w_norms = {}
        self.g_norms = {}

    def __call__(self, model: nn.Module, rank: int, where: float):
        self.current_step += 1
        for name, param in model.named_parameters():
            self.w_norms[name] = self.w_norms.get(name, 0) + param.norm().cpu().item()
            if param.grad is not None:
                self.g_norms[name] = (
                    self.g_norms.get(name, 0) + param.grad.norm().cpu().item()
                )

        if self.current_step >= self.accum_steps:
            self.current_step = 0
            self._dump_stats(rank=rank, where=where)
            self.w_norms = {}
            self.g_norms = {}

    def _dump_stats(self, rank: int, where: float):
        norms = {k: v / float(self.accum_steps) for k, v in self.w_norms.items()}
        grads = {k: v / float(self.accum_steps) for k, v in self.g_norms.items()}
        stats = {
            "where": where,
            "rank": rank,
            "max_norm": max(norms.values()),
            "min_norm": min(norms.values()),
            "max_grad": max(grads.values()),
            "min_grad": min(grads.values()),
        }
        if self.detailed:
            stats.update(
                {
                    "norms": norms,
                    "grads": grads,
                }
            )

        file_name, file_ext = os.path.splitext(self.output_path)
        rank_output_path = f"{file_name}_{rank}{file_ext}"
        with g_pathmgr.open(rank_output_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
