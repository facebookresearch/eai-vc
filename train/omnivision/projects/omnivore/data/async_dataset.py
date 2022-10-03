from typing import Any, Dict, Iterable

from omnivore.data.async_dataset_helpers_fairvit import (
    AsyncDataset,
    AsyncToIterableDataset,
)

from omnivore.data.torch_dataset import TorchDataset
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class AsyncDatasetWrapper(AsyncDataset[Dict[str, Any]]):
    def __init__(self, dataset: data.Dataset):
        """
        Args:
            dataset: a map-style (indexable) dataset. It must be cheap to index.
                For example it can be a `DatasetFromList`.
                Each element must be a Dict[str, Any].
            read_keys: mapping from the dictionary key that contains manifold path
                to the new dictionary key to save read manifold data, e.g. {"file_name": "image"}.
                The output dictionary will contain a new field "image", with data read from field "file_name".
                The data is represented as raw bytes.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    async def __getitem__(self, idx: int) -> Any:
        value = self.dataset[idx]
        return value


def make_async_and_add_prefetch(
    dataset: data.Dataset, sampler: Sampler, max_prefetch: int = 256
):
    async_dataset = AsyncDatasetWrapper(dataset)
    prefetch_async_dataset = AsyncToIterableDataset(
        async_dataset, sampler, max_prefetch
    )
    return prefetch_async_dataset


class AsyncTorchDataset(TorchDataset):
    def __init__(self, max_prefetch: int = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_dataset = self.dataset
        self.max_prefetch = max_prefetch

    def get_loader(self, epoch) -> Iterable:
        self.sampler.set_epoch(epoch)
        self.dataset = make_async_and_add_prefetch(
            self.orig_dataset, self.sampler, self.max_prefetch
        )

        # now we make a DataLoader
        # this will not take in a `Sampler` since self.dataset is an IterableDataset now
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
