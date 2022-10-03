import torch.nn as nn
from omnivore.losses import CORE_LOSS_KEY


class ScaledLoss(nn.Module):
    def __init__(self, loss_fn, scale):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = scale

    def forward(self, *args, **kwargs):
        loss = self.loss_fn(*args, **kwargs)
        if isinstance(loss, dict):
            loss[CORE_LOSS_KEY] *= self.scale
        else:
            loss *= self.scale
        return loss
