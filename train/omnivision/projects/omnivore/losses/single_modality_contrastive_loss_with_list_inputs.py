from typing import List

import torch
from omnivore.losses.clip_loss import CLIPLoss
from omnivore.utils.distributed import all_gather_batch


class SingleModalityContrastiveLossWithListInputs(CLIPLoss):
    def __init__(
        self,
        feat_name: str,
        detach_probs: List[int],
        logit_scale_value: float,
        all_gather_fn: callable = all_gather_batch,
        normalize: bool = True,
        loss1_weight: float = 0.5,
        loss2_weight: float = 0.5,
    ):
        super().__init__(all_gather_fn, normalize, loss1_weight, loss2_weight)
        self.feat_name = feat_name
        self.detach_probs = detach_probs
        self.logit_scale_value = logit_scale_value
        all_zeros = True
        for prob in self.detach_probs:
            assert prob >= 0.0 and prob <= 1.0
            all_zeros = all_zeros and (all_zeros == 0)
        assert all_zeros is False, "Detaching all inputs?!"

    def forward(self, outputs, labels):
        del labels
        feats = outputs[self.feat_name]
        assert isinstance(feats, list)
        assert (
            len(feats) == 2
        ), f"More than 2 crops not supported yet. Got {len(feats)} features"
        assert len(self.detach_probs) == len(feats)

        for idx in range(len(feats)):
            if self.detach_probs[idx] > torch.rand(1).item():
                feats[idx] = feats[idx].detach()

        new_outputs = {
            "image_embed": feats[0],
            "text_embed": feats[1],
            "logit_scale": self.logit_scale_value,
        }
        return super().forward(new_outputs)
