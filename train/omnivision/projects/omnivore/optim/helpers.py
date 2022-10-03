import torch.nn as nn
from omnivore.models.fsdp_model_utils import clip_fsdp_gradients, is_fsdp


class GradientClipper:
    """
    Gradient clipping utils that works for both FSDP and DDP
    """

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2):
        assert isinstance(max_norm, (int, float)) or max_norm is None
        self.max_norm = max_norm if max_norm is None else float(max_norm)
        self.norm_type = norm_type

    def __call__(self, model: nn.Module):
        if self.max_norm is None:
            return  # no-op
        if is_fsdp(model):
            clip_fsdp_gradients(model, max_norm=self.max_norm, norm_type=self.norm_type)
        else:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
            )
