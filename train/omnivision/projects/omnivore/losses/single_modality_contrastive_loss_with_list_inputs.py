from typing import List

import torch
from omnivore.losses import BaseLoss
from omnivore.losses.clip_loss import CLIPLoss
from omnivore.utils.distributed import all_gather_batch


class SingleModalityContrastiveLossWithListInputs(BaseLoss):
    def __init__(
        self,
        feat_name: str,
        detach_probs: List[int],
        logit_scale_value: float,
        all_gather_fn: callable = all_gather_batch,
        normalize: bool = True,
        loss1_weight: float = 0.5,
        loss2_weight: float = 0.5,
        mask_with_data_valid: bool = False,
    ):
        super().__init__()
        self.clip_loss = CLIPLoss(
            all_gather_fn=all_gather_fn,
            normalize=normalize,
            loss1_weight=loss1_weight,
            loss2_weight=loss2_weight,
            mask_with_data_valid=mask_with_data_valid,
        )
        self.feat_name = feat_name
        self.detach_probs = detach_probs
        self.logit_scale_value = logit_scale_value
        self.mask_with_data_valid = mask_with_data_valid
        all_zeros = True
        for prob in self.detach_probs:
            assert prob >= 0.0 and prob <= 1.0
            all_zeros = all_zeros and (all_zeros == 0)
        assert all_zeros is False, "Detaching all inputs?!"

    def core_forward(self, outputs, sample):
        feats = outputs[self.feat_name]
        assert isinstance(feats, list), f"Expect a list, got {type(feats)}"
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

        if self.mask_with_data_valid:
            new_outputs["data_valid"] = sample.data_valid

        return self.clip_loss(new_outputs)
