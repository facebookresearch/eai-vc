# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict

import torch
from omnivision.meters.omnivision_meter import OmnivisionMeter
from omnivision.utils.generic import maybe_convert_to_one_hot

from omnivore.meters.avg_precision_utils import get_precision_recall


class MeanAvgPrecision(OmnivisionMeter):
    """Computes mAP"""

    def __init__(
        self, *args, prec_recall_fn: callable = get_precision_recall, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prec_recall_fn = prec_recall_fn
        self.register_buffer("targets", [], None)
        self.register_buffer("scores", [], None)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the state with predictions and target.
        Args:
            preds: tensor of shape (B, C) where each value is either logit or
                class probability.
            target: tensor of shape (B, C), which contains number of
                instances of class c in element i.
        """
        # Convert target to 0/1 encoding if isn't
        target = maybe_convert_to_one_hot(target, preds)

        assert preds.shape == target.shape, (
            "predictions and target must be of the same shape. "
            f"Got preds({preds.shape}) vs target({target.shape})."
        )
        self.scores.append(preds.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.targets) == 0:
            assert len(self.scores) == 0
            return {"": torch.nan}

        all_targets = torch.vstack(self.targets)
        all_scores = torch.vstack(self.scores)
        num_classes = all_scores.size(1)
        AP = torch.ones((num_classes,))
        AP[:] = torch.nan
        for c in range(num_classes):
            this_target = all_targets[:, c].numpy()
            if sum(this_target) > 0:
                # At least 1 positive
                # Convert to float in case it's in bfloat16 etc.
                _, _, _, ap = self.prec_recall_fn(
                    this_target, all_scores[:, c].float().numpy()
                )
                AP[c] = ap[0]
        return {"": torch.mean(AP).item() * 100.0}
