# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from omnivision.meters.omnivision_meter import OmnivisionMeter


class CrossModalityRetrieval(OmnivisionMeter):
    """Retrieves one feature given another feature, and computes recall@N based
    on retrieval quality.
    """

    def __init__(
        self,
        query_feature: str,
        corpus_feature: str,
        *args,
        topks: Sequence[int] = (1, 10),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_feature_name = query_feature
        self.corpus_feature_name = corpus_feature
        self.topks = topks
        self.num_queries = None  # Set in the update fn
        self.register_buffer("query_features", [], None)
        self.register_buffer("corpus_features", [], None)

    def update(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        del target  # Don't need this for retrieval metrics
        query_feature = preds[self.query_feature_name]
        if isinstance(query_feature, List):
            nq = len(query_feature)
            query_feature = torch.cat([el.unsqueeze(1) for el in query_feature], dim=1)
        else:
            nq = 1
            # Add a singleton dimension so it matches the list above
            query_feature = query_feature.unsqueeze(1)
        assert self.num_queries is None or self.num_queries == nq, (
            "Checks that same num queries for each audio. "
            "The metric doesn't need this constraint but overall code does "
            "for batching. "
            "Helps catch no mean or mergeing of captions features was done."
        )
        self.num_queries = nq
        self.query_features.append(query_feature.detach().cpu())
        corpus_feature = preds[self.corpus_feature_name]
        if isinstance(corpus_feature, List):
            assert len(corpus_feature) == 1, "Only 1 answer to given queries"
            corpus_feature = corpus_feature[0]
        self.corpus_features.append(corpus_feature.detach().cpu())

    def compute(self) -> Dict[str, torch.Tensor]:
        if len(self.query_features) == 0:
            assert len(self.corpus_features) == 0
            return {}
        all_query_features = torch.vstack(self.query_features)
        all_corpus_features = torch.vstack(self.corpus_features)
        assert all_query_features.size(0) == all_corpus_features.size(
            0
        ), f"{all_query_features.shape} {all_corpus_features.shape}"
        true_mapping = torch.arange(start=0, end=all_query_features.size(0))
        num_queries_per_corpus = all_query_features.size(1)
        # Remove the middle dimension
        all_query_features = torch.flatten(all_query_features, 0, 1)
        true_mapping = torch.repeat_interleave(true_mapping, num_queries_per_corpus)
        true_mapping = true_mapping.reshape((-1, 1))
        # L2 normalize the features
        all_query_features = nn.functional.normalize(all_query_features, dim=1, p=2)
        all_corpus_features = nn.functional.normalize(all_corpus_features, dim=1, p=2)
        similarity = torch.mm(all_query_features, all_corpus_features.t())
        res = {}
        res["num_query"] = int(similarity.size(0))
        res["num_corpus"] = int(similarity.size(1))
        for k in self.topks:
            _, indices = similarity.topk(k, largest=True, sorted=True)
            matches = indices == true_mapping
            recall = torch.mean(torch.max(matches, dim=-1)[0].float()) * 100.0
            # Adding the num queries to the meter name because I once made the
            # mistake of averaging all text features into 1 and then doing the
            # evaluation. This will ensure that the num queries is front and
            # center when someone looks at the meter.
            # Eg. Make sure to set list input reduction to "no_op" if using
            # multimodal_wrapper.
            res[f"recall@{k}_{self.num_queries}queriesPerCorpus"] = recall.item()
        return res
