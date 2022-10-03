# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from https://github.com/facebookresearch/recipes/blob/main/torchrecipes/vision/image_classification/metrics/multilabel_accuracy.py
# but includes a bugfix - torchrecipes cast the target to an int, resulting in all zeros!
# Based on https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/meters/accuracy_meter.py.

import enum
from typing import Dict

import torch
from omnivision.meters.omnivision_meter import OmnivisionMeter
from omnivision.utils.generic import maybe_convert_to_one_hot
from torch.distributed import ReduceOp


class MultilabelModes(enum.Enum):
    CLASSY = "classy"
    RECALL = "recall"


class AccuracyMeter(OmnivisionMeter):
    """Computes top-k accuracy for multilabel targets. A sample is considered
    correctly classified if the top-k predictions contain any of the labels.

    Args:
        top_k: Number of highest score predictions considered to find the
            correct label.
    """

    def __init__(
        self, top_k: int, multilabel_mode: str = MultilabelModes.CLASSY
    ) -> None:
        super().__init__()
        self._top_k = top_k
        self._multilabel_mode = MultilabelModes(multilabel_mode)
        self.register_buffer("correct", torch.tensor(0.0), ReduceOp.SUM)
        self.register_buffer("total", torch.tensor(0.0), ReduceOp.SUM)

    @staticmethod
    def compute_correct_total(preds, target, topk, multilabel_mode):
        """
        Args:
            preds: Tensor (N, C): N vectors with distribution over C classes
            target: Tensor (N, C): one-hot or multi-hot vector of labels.
        """
        # If Pytorch AMP is being used, model outputs are probably fp16
        # Since .topk() is not compatible with fp16, we promote the model outputs to
        # full precision
        _, top_idx = preds.float().topk(topk, dim=1, largest=True, sorted=True)
        topk_matches = torch.gather(target, dim=1, index=top_idx)
        # TODO Need to add this assertion, however it breaks
        # when we run the meter on training data where there
        # might be smoothing/cutmix/mixup etc.
        # assert torch.all(
        #     torch.logical_or(target == 0, target == 1)
        # ), "Must be a 0-1 tensor"
        if multilabel_mode == MultilabelModes.CLASSY:
            # In multilabel cases this will compute a "precision"-like metric,
            # where it's "correct" if any of the positives are in the topk.
            correct = topk_matches.max(dim=1).values.sum().item()
            total = target.size(0)
        elif multilabel_mode == MultilabelModes.RECALL:
            # Here we compute in top-K, how many positive classes did we get
            # out of of the total positive classes we were supposed to get
            correct = topk_matches.sum().item()
            total = target.sum().item()
        else:
            raise NotImplementedError(multilabel_mode)
        return torch.tensor(correct), torch.tensor(total)

    @staticmethod
    def compute_accuracy(correct, total):
        if torch.is_nonzero(total):
            return (correct / total).item() * 100
        return 0.0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the state with predictions and target.
        Args:
            preds: tensor of shape (B, C) where each value is either logit or
                class probability.
            target: tensor of shape (B, C), which is one-hot / multi-label
                encoded.
        """
        # Convert target to 0/1 encoding if isn't
        target = maybe_convert_to_one_hot(target, preds)

        assert preds.shape == target.shape, (
            "predictions and target must be of the same shape. "
            f"Got preds({preds.shape}) vs target({target.shape})."
        )
        num_classes = target.shape[1]
        assert (
            num_classes >= self._top_k
        ), f"top-k({self._top_k}) is greater than the number of classes({num_classes})"
        correct, total = self.compute_correct_total(
            preds, target, self._top_k, self._multilabel_mode
        )
        self.correct += correct
        self.total += total

    def compute(self) -> Dict[str, torch.Tensor]:
        return {"": self.compute_accuracy(self.correct, self.total)}
