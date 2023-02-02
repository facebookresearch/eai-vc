# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Optional

import numpy as np
import torch
from omnivore.losses import BaseLoss
from omnivore.losses.clip_loss import CLIPLoss
from omnivore.utils.distributed import all_gather_batch


class ContrastiveLoss(BaseLoss):
    """
    Wrapper around the CLIP loss
    Expects the logit scale to be provided in the model outputs
    This allows us to have learned logit scale, i.e., "temperature" parameters that live in the model
    Learnable parameters are at one place which makes it easy to optimize/checkpoint
    Args:
        feat1_no_grad: Probability of the feat1 to be detached and no gradient to be passed back.
    """

    def __init__(
        self,
        feat1_name: str,
        feat2_name: str,
        feat1_no_grad: float = 0.0,
        feat2_no_grad: float = 0.0,
        logit_scale_name: Optional[str] = None,
        all_gather_fn: callable = all_gather_batch,
        max_temperature_multiplier: float = 100,
        normalize: bool = True,
        loss1_weight: float = 0.5,
        loss2_weight: float = 0.5,
        label_smoothing: float = 0.0,
        mask_with_data_valid: bool = False,
    ):
        super().__init__()
        self.clip_loss = CLIPLoss(
            all_gather_fn=all_gather_fn,
            normalize=normalize,
            loss1_weight=loss1_weight,
            loss2_weight=loss2_weight,
            label_smoothing=label_smoothing,
            mask_with_data_valid=mask_with_data_valid,
        )
        self.max_temperature_multiplier = max_temperature_multiplier
        self.feat1_name = feat1_name
        self.feat2_name = feat2_name
        self.logit_scale_name = logit_scale_name
        self.feat1_no_grad = feat1_no_grad
        self.feat2_no_grad = feat2_no_grad
        self.mask_with_data_valid = mask_with_data_valid
        assert self.feat1_no_grad >= 0.0 and self.feat1_no_grad <= 1.0
        assert self.feat2_no_grad >= 0.0 and self.feat2_no_grad <= 1.0
        assert not (
            self.feat1_no_grad and self.feat2_no_grad
        ), "Both features without grad?!"

    def core_forward(self, outputs, sample):
        new_outputs = {}
        feat1 = outputs[self.feat1_name]
        feat2 = outputs[self.feat2_name]
        if self.feat1_no_grad > torch.rand(1).item():
            feat1 = feat1.detach()
        if self.feat2_no_grad > torch.rand(1).item():
            feat2 = feat2.detach()
        new_outputs["image_embed"] = feat1
        new_outputs["text_embed"] = feat2
        if self.logit_scale_name is not None:
            new_outputs["logit_scale"] = torch.clip(
                outputs[self.logit_scale_name], max=self.max_temperature_multiplier
            )
        else:
            new_outputs["logit_scale"] = 1.0  # no-op
        if self.mask_with_data_valid:
            new_outputs["data_valid"] = sample.data_valid
        return self.clip_loss(new_outputs)

    def extra_repr(self):
        return f"feat1_name={self.feat1_name}, feat2_name={self.feat2_name}, feat1_no_grad={self.feat1_no_grad}, feat2_no_grad={self.feat2_no_grad}, logit_scale_name={self.logit_scale_name}, all_gather_fn={self.clip_loss.all_gather_fn}, max_temperature_multiplier={self.max_temperature_multiplier}, normalize={self.clip_loss.normalize}, loss1_weight={self.clip_loss.loss1_weight}, loss2_weight={self.clip_loss.loss2_weight}, mask_with_data_valid={self.clip_loss.mask_with_data_valid}"


class ContrastiveLossLegacy(CLIPLoss):
    def __init__(
        self,
        feat1_name: str,
        feat2_name: str,
        temperature: float = 0.07,
        all_gather_fn: callable = all_gather_batch,
        learnable_temperature: bool = False,
        max_temperature_multiplier: float = 100,
    ):
        super().__init__(all_gather_fn=all_gather_batch)
        logit_scale_data = torch.ones([], dtype=torch.float32) * np.log(1 / temperature)
        if learnable_temperature:
            self.logit_scale = torch.nn.Parameter(logit_scale_data, requires_grad=True)
        else:
            self.register_buffer("logit_scale", logit_scale_data)
        self.max_temperature_multiplier = max_temperature_multiplier
        self.feat1_name = feat1_name
        self.feat2_name = feat2_name

    def forward(self, outputs, labels):
        del labels  # This SSL loss doesn't need labels
        new_outputs = {}
        new_outputs["image_embed"] = outputs[self.feat1_name]
        new_outputs["text_embed"] = outputs[self.feat2_name]
        logit_scale = self.logit_scale.exp()
        if self.max_temperature_multiplier is not None:
            logit_scale = torch.clip(logit_scale, max=self.max_temperature_multiplier)
        new_outputs["logit_scale"] = logit_scale
        return super().forward(new_outputs)["loss"]
