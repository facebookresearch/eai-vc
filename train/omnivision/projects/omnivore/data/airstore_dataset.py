#!/usr/bin/env python3

import io
import logging
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Callable, Iterable, Optional, Tuple

import torch.utils.data

try:
    from airstore.client.airstore_tabular import AIRStorePathHandler
except ImportError:
    logging.warn("Airstore client is not supported in this environment.")
except Exception as e:
    logging.warn(f"Encountered error when importing Airstore: {e}")

from iopath.common.file_io import PathManager
from omnivision.utils.distributed import get_rank, get_world_size
from omnivore.data.api import VisionSample, VisionTextSample
from omnivore.utils.data import get_mean_image
from PIL import Image

from torch.utils.data import DataLoader, Dataset

from .omni_dataset import OmniDataset

from .path_dataset import DEFAULT_SPATIAL_SIZE


class AirStoreTorchDataLoader(OmniDataset):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self._phased_epoch = 0
        self.dataset.set_batch_size(self.batch_size)

    def get_loader(self, epoch) -> Iterable:
        self.dataset.set_epoch(epoch)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=None,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )


class AIRStoreDataset(torch.utils.data.IterableDataset, ABC):
    """
    AIRStore dataset.
    Dataset is sharded into airstore_world_size (global number of data
    loading workers) number of shards. Samples not a multiple of
    airstore_world_size are discarded.
    """

    def __init__(
        self,
        table_name: str,
        total_length: int,
        transforms: Any,
        data_column: str,
        label_column: str,
        phases_per_epoch: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        limit: int = -1,
        id_column: Optional[str] = None,
    ):
        """
        Args:
            table_name: Airstore table name.
            transforms: Omnivison-style list of transfroms.
            data_column: Name of the column with the images/videos.
            label_column: Name of the column with labels.
            id_column: Name of the column with id's for each sample.
            phases_per_epoch: How many training epochs to spread the data over.
            drop_last: Drop incomplete batches towards the end of the training epoch.
            shuffle: If enabled, shuffles the local fetched batch.
            limit: -1 means disabled. If set, limits the entire training data
                to the limit number of samples.

        """
        super().__init__()
        self._phased_epoch = 0
        self._table_name = table_name
        self._phases_per_epoch = phases_per_epoch
        self._total_length = total_length
        self._pathmgr = PathManager()
        self._pathmgr.register_handler(AIRStorePathHandler())
        self._pathmgr.set_strict_kwargs_checking(False)
        self._torch_dataloader_length = None
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._batch_size = None

        self._transforms = transforms
        self._data_column = data_column
        self._id_column = id_column
        self._label_column = label_column

    def set_epoch(self, epoch: int) -> None:
        self._phased_epoch = epoch

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def apply_transforms(self, sample):
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def __iter__(self):
        """
        Returns an iterator for (transformed) image and label
        """
        assert (
            self._batch_size is not None
        ), "Please set the batch_size using set_batch_size method in Airstore dataset."
        world_size = get_world_size()
        rank = get_rank()
        num_workers, worker_id = self.get_worker_info()

        # dataset sharded across airstore_world_size workers
        airstore_world_size = world_size * num_workers
        # each worker takes it's shard based on parent process rank and worker id
        airstore_rank = rank * num_workers + worker_id

        # compute skip samples. This is computed per worker.
        skip_samples = (len(self) // num_workers) * (
            self._phased_epoch % self._phases_per_epoch
        )
        actual_epoch = self._phased_epoch // self._phases_per_epoch

        logging.info(
            "######## "
            + f"Creating Airstore dataloader with Airstore Rank: {airstore_rank}, "
            + f"Airstore World Size: {airstore_world_size}, Seed: {actual_epoch}, Phased Epoch: {self._phased_epoch}, "
            + f"Skip Samples: {skip_samples}, #Phases: {self._phases_per_epoch} ########"
        )

        # TODO: Expose limit and offset methods in OpenT, etc in constructor.
        airstore_iter = self._pathmgr.opent(
            f"airstore://{self._table_name}",
            world_size=airstore_world_size,
            rank=airstore_rank,
            enable_shuffle=self._shuffle,
            seed=actual_epoch + 1,
            skip_samples=skip_samples,
            env="rsc",
        )
        return map(lambda sample: self._map_item(sample), airstore_iter)

    @abstractmethod
    def _map_item(self, sample: Any) -> Any:
        pass

    def get_worker_info(self) -> Tuple[int, int]:
        num_workers = 1
        worker_id = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        return max(num_workers, 1), worker_id

    def get_total_length(self):
        return self._total_length

    def __len__(self):
        """
        Intended for use *only* by torch.utils.data.DataLoader;
        If total len of the dataset is desired, use get_total_length instead.
        """
        if self._torch_dataloader_length:
            return self._torch_dataloader_length

        # Samples discarded by sharding
        world_size = get_world_size()
        num_workers, _ = self.get_worker_info()
        num_samples_per_worker = self.get_total_length() // (num_workers * world_size)

        # Samples discarded due to drop_last=True. In map-style
        # datasets this setting typically drops at most one batch
        # across workers, due to sampler's assignment of indices.
        # Here, however, each worker has the same shard size, so
        # the last batch is dropped from every worker.
        if self._drop_last:
            num_batches_per_worker = num_samples_per_worker // self._batch_size
        else:
            num_batches_per_worker = ceil(num_samples_per_worker / self._batch_size)

        num_batches_per_worker = num_batches_per_worker // self._phases_per_epoch

        return num_batches_per_worker * self._batch_size * num_workers


def default_img_generator():
    return get_mean_image(DEFAULT_SPATIAL_SIZE)


class AirstoreImageDataset(AIRStoreDataset):
    def _map_item(self, sample):

        try:
            image = Image.open(io.BytesIO(sample[self._data_column]))
            if image.mode != "RGB":
                image = image.convert("RGB")
            out_sample = VisionSample(
                vision=image,
                label=sample[self._label_column],
                data_idx=int(sample[self._id_column]) if self._id_column else -1,
                data_valid=True,
            )
            out_sample = self.apply_transforms(out_sample)
        except Exception as e:
            logging.warn(
                f"############## Airstore Image Loading Error: {e} ###################"
            )
            image = default_img_generator()
            out_sample = VisionSample(
                vision=image,
                label=sample[self._label_column],
                data_idx=int(sample[self._id_column]) if self._id_column else -1,
                data_valid=False,
            )
            out_sample = self.apply_transforms(out_sample)
        return out_sample


class AirstoreImageTextDataset(AIRStoreDataset):
    def __init__(self, text_column: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._text_column = text_column

    def _map_item(self, sample):

        try:
            image = Image.open(io.BytesIO(sample[self._data_column]))
            if image.mode != "RGB":
                image = image.convert("RGB")
            out_sample = VisionTextSample(
                vision=image,
                label=sample[self._label_column],
                data_idx=int(sample[self._id_column]) if self._id_column else -1,
                text=sample[self._text_column],
                data_valid=True,
            )
            out_sample = self.apply_transforms(out_sample)
        except Exception as e:
            logging.warn(
                f"############## Airstore Image Loading Error: {e} ###################"
            )
            image = default_img_generator()
            out_sample = VisionTextSample(
                vision=image,
                label=sample[self._label_column],
                data_idx=int(sample[self._id_column]) if self._id_column else -1,
                text=sample[self._text_column],
                data_valid=False,
            )
            out_sample = self.apply_transforms(out_sample)
        return out_sample
