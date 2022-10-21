import ast
import json
import logging
import math
import os
import random
import sys
from multiprocessing import Value
from typing import Callable, Iterable, List

import numpy as np
import torch

import webdataset as wds

from omnivision.utils.distributed import get_rank

from omnivore.data.api import BatchVisionTextSample, VisionTextSample
from omnivore.data.omni_dataset import OmniDataset
from omnivore.data.vision_text_dataset import get_default_text_string
from omnivore.utils.data import get_mean_image
from torch.utils.data import Dataset, get_worker_info, IterableDataset
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


class WebVisionTextSimple(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        shuffle: int,
        text_key: str,
        decode_str: str,
        to_tuple: List[str],
        base_dataset_length: int,
    ) -> None:
        """
        A simple wrapper for WebDataset
        Args
        - base_dataset: a dataset that returns the a tuple of visual data and text
        - shuffle: integer value specifying the buffer size to use for shuffling.
        - text_key: a text key. Can be set to None
        - decode_str: a string indicating decoding for the visual_data
            See https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py#L175
        - to_tuple: the associated files to read
        - base_dataset_length: specified dataset length
        """
        self.base_dataset_length = int(base_dataset_length)
        self.base_dataset = (
            base_dataset.shuffle(shuffle)
            .decode(decode_str)
            .to_tuple(*to_tuple)
            .with_epoch(self.base_dataset_length)
        )
        self.text_key = text_key
        self.base_dataset_iterator = None

    def __getitem__(self, idx):
        # idx is ignored since WebDatasets are IterableDatasets
        if self.base_dataset_iterator is None:
            self.base_dataset_iterator = iter(self.base_dataset)
        try:
            visual_data, text = next(self.base_dataset_iterator)
            if self.text_key is not None:
                text = text[self.text_key]
            data_valid = True
        except Exception:
            # failed to decode image/text
            visual_data = None
            text = None
            data_valid = False

        if visual_data is None:
            data_valid = False
            visual_data = get_mean_image(224)

        if text is None:
            data_valid = False
            text = get_default_text_string()
        # webdataset doesn't support indexing, so data_idx=-1
        # webdataset loads free-form text, so label=-1
        return VisionTextSample(
            vision=visual_data, data_idx=-1, label=-1, data_valid=data_valid, text=text
        )

    def __len__(self):
        if not hasattr(self.base_dataset, "__len__"):
            return self.base_dataset_length
        return len(self.base_dataset)


class WebVisionTextPipeline(Dataset):
    def __init__(
        self,
        base_dataset_fn: Callable,
        base_dataset_length: int,
        epoch: int = 0,
    ) -> None:
        """
        A simple wrapper for WebDataset using the Pipeline interface
        Args
        - base_dataset_fn: a function that returns the dataset. Accepts a `epoch` value which is a multiprocessing Value
        - base_dataset_length: specified dataset length
        """
        self.base_dataset_length = int(base_dataset_length)
        shared_epoch = SharedEpoch(
            epoch=epoch
        )  # create a shared epoch store to sync epoch to dataloader worker proc
        self.base_dataset = base_dataset_fn(epoch=shared_epoch)
        self.base_dataset_iterator = None
        self.shared_epoch = shared_epoch

    def __getitem__(self, idx):
        # idx is ignored since WebDatasets are IterableDatasets
        if self.base_dataset_iterator is None:
            epoch_val = self.shared_epoch.get_value()
            self.shared_epoch.set_value(epoch_val + 1)
            self.base_dataset_iterator = iter(self.base_dataset)
        try:
            visual_data, text = next(self.base_dataset_iterator)
            data_valid = True
        except Exception:
            # failed to decode image/text
            visual_data = None
            text = None
            data_valid = False

        if visual_data is None:
            data_valid = False
            visual_data = get_mean_image(224)

        if text is None:
            data_valid = False
            text = get_default_text_string()
        # webdataset doesn't support indexing, so data_idx=-1
        # webdataset loads free-form text, so label=-1
        return VisionTextSample(
            vision=visual_data, data_idx=-1, label=-1, data_valid=data_valid, text=text
        )

    def __len__(self):
        if not hasattr(self.base_dataset, "__len__"):
            return self.base_dataset_length
        return len(self.base_dataset)


class WebVisionDatasetBatchedWithLoader(OmniDataset):
    def __init__(
        self,
        base_dataset_fn: Callable,
        base_loader_fn: Callable,
        epoch: int = 0,
        num_workers: int = 0,
    ) -> None:
        # num_workers is ignored.
        # base dataset/loader_fn must have this.
        shared_epoch = SharedEpoch(
            epoch=epoch
        )  # create a shared epoch store to sync epoch to dataloader worker proc
        self.shared_epoch = shared_epoch
        self.dataset = base_dataset_fn(shared_epoch=shared_epoch)
        self.base_loader_fn = base_loader_fn

    def get_loader(self, epoch) -> Iterable:
        self.shared_epoch.set_value(epoch)
        return self.base_loader_fn(dataset=self.dataset)


### Helper functions and interface from https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
### Pipeline interface for webdataset


def get_dataset_size(shards_list):
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def filter_no_caption(sample):
    return "txt" in sample


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    # worker_info = get_worker_info()
    # if worker_info is not None:
    # favour the seed already created for pytorch dataloader workers if it exists
    # return worker_info.seed
    # fallback to wds rank based seed
    return (
        wds.utils.pytorch_worker_seed()
    )  # Preferred since provides 0 based seeds (e.g. 0, 1, 2, 3)


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed((self.worker_seed() * 1e6) + (get_rank() * 1e3) + epoch)
        for _ in range(self.nshards):
            yield {"url": self.rng.choice(self.urls)}


def get_wds_dataset(
    urls,
    epoch=0,
    resampled=False,
    seed=42,
    shard_shuffle_size=2000,
    shard_shuffle_initial=500,
    sample_shuffle_size=5000,
    sample_shuffle_initial=1000,
):
    if resampled:
        pipeline = [ResampledShards2(urls, deterministic=True, epoch=epoch)]
    else:
        pipeline = [wds.SimpleShardList(urls)]

    # at this point we have an iterator over all the shards

    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=shard_shuffle_size,
                    initial=shard_shuffle_initial,
                    seed=seed,
                    epoch=epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=sample_shuffle_size,
                initial=sample_shuffle_initial,
            ),
        ]
    )
    pipeline.extend(
        [
            wds.select(filter_no_caption),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", text="txt"),
            wds.to_tuple("image", "text"),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    return dataset


# Follows https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py#L290
def get_wds_dataset_batched(
    urls,
    shared_epoch,
    dataset_size_file,
    batch_size,
    num_workers,
    preprocess_img,
    preprocess_txt,
    seed=42,
    is_train=True,
    resampled=False,
    shard_shuffle_size=2000,
    shard_shuffle_initial=500,
    sample_shuffle_size=5000,
    sample_shuffle_initial=1000,
    floor=False,
):
    dt = np.load(dataset_size_file)
    num_samples = dt.sum()
    num_shards = len(dt)
    assert isinstance(shared_epoch, SharedEpoch)
    if resampled:
        pipeline = [ResampledShards2(urls, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(urls)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=shard_shuffle_size,
                        initial=shard_shuffle_initial,
                        seed=seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=sample_shuffle_size,
                    initial=sample_shuffle_initial,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png", text="txt"),
            wds.map_dict(image=preprocess_img, text=preprocess_txt),
            wds.to_tuple("image", "text"),
            wds.batched(batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        world_size = torch.distributed.get_world_size()
        if not resampled:
            assert (
                num_shards >= num_workers * world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, num_workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
        dataset.num_worker_batches = num_worker_batches
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)
    dataset.num_batches = num_batches
    return dataset


class WebLoaderWrapper(wds.WebLoader):
    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.num_batches = dataset.num_batches

    def __len__(self):
        return self.num_batches


def get_wds_loader(
    dataset,
    num_workers,
    **kwargs,
):
    dataloader = WebLoaderWrapper(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        **kwargs,
    )
    return dataloader


class BatchToSampleText(Callable):
    def __init__(self, collate_fn: Callable) -> None:
        super().__init__()
        self.collate_fn = collate_fn

    def __call__(self, batch_in):
        assert len(batch_in) == 2
        data = batch_in[0]
        text = batch_in[1]
        data_valid = torch.ones(data.shape[0])
        data_idx = torch.ones(data.shape[0]) * -1
        data_idx = torch.ones(data.shape[0]) * -1
        singleton_batch = BatchVisionTextSample(
            vision=data,
            text=text,
            data_idx=data_idx,
            data_valid=data_valid,
            label=data_idx,
        )
        return self.collate_fn(singleton_batch)
