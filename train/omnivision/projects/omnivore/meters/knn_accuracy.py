# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Dict, Optional, Sequence

import scipy.spatial
import torch
from omnivision.meters.accuracy_meter import AccuracyMeter, MultilabelModes
from omnivision.meters.omnivision_meter import OmnivisionMeter
from omnivision.utils.generic import convert_to_one_hot, maybe_convert_to_one_hot
from omnivore.meters import FINAL_LOGITS_NAME


class KnnAccuracy(OmnivisionMeter):
    """Performs retrieval on some feature, and computes KNN based accuracy
    (or recall@k, in case of multilabel datasets).
    Note that this is doing retrieval over the validation features only,
    so this is just a debuggin metric useful for evaluating if the
    feature space learned is useful or not. The correct number would be
    if the retrieval was done on the training data.
    """

    def __init__(
        self,
        feat_name: str,
        *args,
        topks: Sequence[int] = (10,),
        temperature: float = 0.1,
        num_classes: Optional[int] = None,
        multilabel_mode: str = MultilabelModes.CLASSY,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feat_name = feat_name
        self.topks = topks
        self.temperature = temperature
        self.num_classes = num_classes
        self.multilabel_mode = MultilabelModes(multilabel_mode)
        self.register_buffer("features", [], None)
        self.register_buffer("targets", [], None)

    def update(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        """Updates the state with predictions and target.
        Args:
            preds: Feature dictionary tensor of shape (B, C) where each value is either logit or
                class probability.
            target: tensor of shape (B, C), which contains number of
                instances of class c in element i.
        """
        # Convert target to 0/1 encoding if isn't
        if target.ndim != 2:
            if FINAL_LOGITS_NAME in preds:
                target = maybe_convert_to_one_hot(target, preds[FINAL_LOGITS_NAME])
                logging.warning(
                    f"Using num_classes={self.num_classes} inferred from logits"
                )
                assert self.num_classes is None or target.size(1) == self.num_classes
            elif self.num_classes is not None:
                assert (
                    isinstance(target, torch.Tensor)
                    and target.ndim == 2
                    and target.size(1) == 1
                ), (
                    "Must be a (N, 1) tensor with each element being the class ID."
                    "Not supported for multi-label, that should have been already converted "
                    "to multi-hot in the collator."
                )
                target = convert_to_one_hot(target, self.num_classes)
            else:
                raise ValueError("Please specify num_classes to create logits")
        preds = preds[self.feat_name]
        self.features.append(preds.detach().cpu())
        self.targets.append(target.detach().cpu())

    @staticmethod
    def features_to_preds(
        features: torch.Tensor,
        labels: torch.Tensor,
        topks: Sequence[int],
        temperature: float,
    ):
        """
        Perform NN retrieval in the feature space and use the top-K NNs to compute
        labels for each sample.
        Inspired from https://github.com/facebookresearch/vissl/blob/64fe10362429ebbeef4912f622d73fb21702c2da/vissl/utils/knn_utils.py#L263
        Args:
            features: (N, D) tensor with D-dim features
            labels: One- or multi-hot representation of the GT (N, C)
        Returns:
            Tensor (N, C) predictions
        """
        batch_size = features.size(0)
        num_classes = labels.size(1)
        assert labels.size(0) == batch_size
        # No cheap way to do pdist in torch (https://github.com/pytorch/pytorch/issues/11202)
        similarity = 1.0 - torch.from_numpy(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(features.numpy(), "cosine")
            )
        )
        # Remove the same element from similarity
        # -1 works since min cosine similarity is 0
        similarity.fill_diagonal_(-1)
        all_preds = {}
        for topk in topks:
            assert topk < (similarity.size(1) - 1), (
                "There need to be enough elements to search topk from (except itself)."
                f"Only found {(similarity.size(1) - 1)}"
            )
            top_similarity, indices = similarity.topk(topk, largest=True, sorted=True)
            top_similarity.div_(temperature).exp_()
            retrieval_one_hot = labels[indices, :]
            all_preds[topk] = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, topk, num_classes),
                    top_similarity.view(batch_size, topk, 1),
                ),
                1,
            )
        return all_preds

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.targets) == 0:
            assert len(self.features) == 0
            return {}
        all_features = torch.vstack(self.features)
        all_targets = torch.vstack(self.targets)
        accs = {}
        all_preds_all_k = self.features_to_preds(
            all_features, all_targets, self.topks, self.temperature
        )
        for k, all_preds in all_preds_all_k.items():
            assert all_preds.shape == all_targets.shape, (
                "predictions and target must be of the same shape. "
                f"Got preds({all_preds.shape}) vs target({all_targets.shape})."
            )
            accs[f"KNN_{k}/top1"] = AccuracyMeter.compute_accuracy(
                *AccuracyMeter.compute_correct_total(
                    all_preds, all_targets, topk=1, multilabel_mode=self.multilabel_mode
                )
            )
            accs[f"KNN_{k}/top5"] = AccuracyMeter.compute_accuracy(
                *AccuracyMeter.compute_correct_total(
                    all_preds, all_targets, topk=5, multilabel_mode=self.multilabel_mode
                )
            )
        return accs
