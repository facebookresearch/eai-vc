import torch.nn as nn
from omnivore.losses import BaseLoss, CORE_LOSS_KEY


class ScaledLoss(BaseLoss):
    def __init__(self, loss_fn, scale):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = scale

    def core_forward(self, *args, **kwargs):
        if hasattr(self.loss_fn, "core_forward"):
            loss = self.loss_fn.core_forward(*args, **kwargs)
        else:
            assert len(args) == 2
            output = args[0]
            label = args[1]
            loss = self.loss_fn(output, label, **kwargs)
        if isinstance(loss, dict):
            loss[CORE_LOSS_KEY] *= self.scale
        else:
            loss *= self.scale
        return loss
