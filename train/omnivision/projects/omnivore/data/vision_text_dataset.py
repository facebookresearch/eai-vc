#!/usr/bin/env python3
import itertools
from abc import ABC
from typing import Any

from torch.utils.data import Dataset, IterableDataset


def get_default_text_string():
    return ""


class VisionTextDataset(Dataset, ABC):
    def __init__(self, base_dataset: Dataset, transforms: Any):
        """
        Args
        - base_dataset: a dataset that returns the input sample and text (raw samples)
        - transform: Transform object that can transform the visual data + tokenize text etc.
        """
        self.base_dataset = base_dataset
        self.transforms = [] if transforms is None else transforms

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self):
        return len(self.base_dataset)


class FilteredVisionTextDataset(IterableDataset, ABC):
    def __init__(self, base_dataset: Dataset, transforms: Any):
        """
        A variant of `VisionTextDataset` which allows transforms to return None
        to skip samples. The dataset cycles back to the start to keep the same
        length. Useful for integration tests.

        Args
        - base_dataset: a dataset that returns the input sample and text (raw samples)
        - transform: Transform object that can transform the visual data + tokenize text etc.
        """
        self.base_dataset = base_dataset
        self.transforms = [] if transforms is None else transforms

    def __iter__(self):
        underlying_len = self.base_dataset
        for idx in itertools.cycle(range(len(underlying_len))):
            sample = self.base_dataset[idx]
            for transform in self.transforms:
                if sample is None:
                    break
                sample = transform(sample)
            if sample is not None:
                yield sample

    def __len__(self):
        return len(self.base_dataset)
