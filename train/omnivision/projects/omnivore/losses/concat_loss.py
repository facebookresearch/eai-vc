from typing import List

import torch
from omnivore.losses import CORE_LOSS_KEY


class ConcatLoss(torch.nn.Module):
    def __init__(self, loss_fns: List[torch.nn.Module]) -> None:
        super().__init__()
        self.loss_fns = torch.nn.ModuleList(loss_fns)

    def update_loss_dict(self, loss_index, loss_dict, loss_out):
        for k in loss_out:
            loss_dict[f"{loss_index}/{k}"] = loss_out[k]
        return loss_dict

    def forward(self, *args, **kwargs):
        loss_val = 0
        loss_dict = {}
        for loss_index, loss in enumerate(self.loss_fns):
            loss_out = loss(*args, **kwargs)
            if isinstance(loss_out, dict):
                core_loss = loss_out[CORE_LOSS_KEY]
                loss_dict = self.update_loss_dict(loss_index, loss_dict, loss_out)
            else:
                core_loss = loss_out
            loss_val += core_loss

        return_value = loss_val
        if len(loss_dict) > 0:
            loss_dict[CORE_LOSS_KEY] = loss_val
            return_value = loss_dict

        return return_value
