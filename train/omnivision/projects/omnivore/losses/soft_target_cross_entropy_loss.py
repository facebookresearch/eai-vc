# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from omnivision.utils.generic import convert_to_one_hot


class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", normalize_targets=True):
        """Intializer for the soft target cross-entropy loss loss.
        This allows the targets for the cross entropy loss to be multilabel
        Args:
            ignore_index: sample should be ignored for loss if the class is this value
            reduction: specifies reduction to apply to the output
            normalize_targets: whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample
        """
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self._ignore_index = ignore_index
        self._reduction = reduction
        assert isinstance(normalize_targets, bool)
        self._normalize_targets = normalize_targets
        if self._reduction not in ["none", "mean"]:
            raise NotImplementedError(
                'reduction type "{}" not implemented'.format(self._reduction)
            )
        self._eps = torch.finfo(torch.float32).eps

    def forward(self, output, target):
        """for N examples and C classes
        - output: N x C these are raw outputs (without softmax/sigmoid)
        - target: N x C or N corresponding targets
        Target elements set to ignore_index contribute 0 loss.
        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """
        # check if targets are inputted as class integers
        if target.ndim == 1:
            assert (
                output.shape[0] == target.shape[0]
            ), "SoftTargetCrossEntropyLoss requires output and target to have same batch size"
            target = convert_to_one_hot(target.view(-1, 1), output.shape[1])
        assert output.shape == target.shape, (
            "SoftTargetCrossEntropyLoss requires output and target to be same "
            f"shape: {output.shape} != {target.shape}"
        )
        valid_mask = target != self._ignore_index
        valid_targets = target.float() * valid_mask.float()
        if self._normalize_targets:
            valid_targets /= self._eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(output, -1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        # perform reduction
        if self._reduction == "mean":
            # normalize based on the number of samples with > 0 non-ignored targets
            loss = per_sample_loss.sum() / torch.sum(
                (torch.sum(valid_mask, -1) > 0)
            ).clamp(min=1)
        elif self._reduction == "none":
            loss = per_sample_loss

        return loss
