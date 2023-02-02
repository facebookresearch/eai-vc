# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import os
import pickle
import random
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Type

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from omnivore.data.api import BatchVisionTextSample, VisionTextSample
from PIL import Image, ImageFile
from torch.utils.data.dataloader import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class Sample:
    data: Any


class DefaultOmnivoreCollator(Callable):
    def __init__(self, wrapper_dataclass: Type = Sample) -> None:
        self.wrapper_dataclass = wrapper_dataclass

    def __call__(self, batch_in):
        batch = []
        for sample in batch_in:
            assert isinstance(sample, self.wrapper_dataclass)
            batch.append(asdict(sample))
        return self.wrapper_dataclass(**default_collate(batch))


class ChunkedCLIPCollator(Callable):
    def __init__(self) -> None:
        self._wrapper_dataclass = Sample

    def __call__(self, batch_in):
        images = []
        captions = []

        for sample in batch_in:
            assert isinstance(sample, self._wrapper_dataclass)
            images += sample.data[0]
            captions += sample.data[1]

        return self._wrapper_dataclass(
            data=(default_collate(images), default_collate(captions))
        )


def yfcc_loader(root, index, cache=None):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2:5]
    file_img = index[5:] + ".jpg"
    if "manifold" in root:
        path_zip = os.path.join(root, repo, z) + ".zip"
    else:
        path_zip = os.path.join(root, "images", repo, z) + ".zip"

    with g_pathmgr.open(path_zip, "rb") as fh:
        with zipfile.ZipFile(fh, "r") as myzip:
            img = copy.deepcopy(Image.open(myzip.open(file_img)))

    return img.convert("RGB")


class YFCCDatasetCLIPChunked(torch.utils.data.Dataset):
    def __init__(
        self,
        root="manifold://omnivore/tree/datasets/yfcc100m/15mil_chunk_64",
        transform=None,
        tokenizer=None,
    ):
        self.transform = transform or (lambda x: x)
        self.tokenizer = tokenizer

        self.root = root

    def __len__(self):
        return 229523  # 14689580(dataset_size) / 64(chunk size)

    def __getitem__(self, i):
        try:
            with g_pathmgr.open(os.path.join(self.root, f"{i}.pkl"), "rb") as f:
                chunk = pickle.load(f)
        except:
            logging.warn(
                f"###################### Failed to fetch chunk at {i} #############"
            )
            return self.__getitem__(random.randint(0, self.__len__()))

        images = chunk["imgs"]
        # apply transformation
        if self.transform is not None:
            images = [self.transform(i) for i in images]

        captions = []
        for sample in chunk["meta_data"]:
            index, title, desc = sample
            caption = np.random.choice([title, desc])
            if self.tokenizer is not None:
                captions.append(self.tokenizer(caption))

        return Sample(data=(images, captions))


class YFCCDatasetCLIPIndividual(torch.utils.data.Dataset):
    def __init__(
        self,
        root="manifold://omnivore/tree/datasets/yfcc100m/unzipped_images_new_split",
        data="manifold://omnivore/tree/datasets/yfcc100m/meta_data/yfcc_meta_data/15m_single_files_new",
        transform=None,
        tokenizer=None,
    ):
        self.transform = transform or (lambda x: x)
        self.tokenizer = tokenizer

        self.root = root
        self.meta_data_root = data

    def __len__(self):
        return 14689580

    def __getitem__(self, i):
        with g_pathmgr.open(os.path.join(self.meta_data_root, f"{i}.pkl"), "rb") as f:
            sample = pickle.load(f)

        index, title, desc = sample
        try:
            index = format(index, "0>8d")
            repo = index[:2]
            z = index[2:5]
            file_img = index[5:] + ".jpg"
            img_path = os.path.join(self.root, repo, z, file_img)

            with g_pathmgr.open(img_path, "rb") as fh:
                img = Image.open(fh).convert("RGB")
        except:
            logging.info(
                f"###################### Failed to fetch image at {img_path} #############"
            )
            return self.__getitem__(random.randint(0, self.__len__()))

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        caption = np.random.choice([title, desc])
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return Sample(data=(img, caption))


class YFCCDatasetCLIP(torch.utils.data.Dataset):
    """
    YFCC dataset for CLIP:
    one view of an image with caption
    """

    def __init__(self, root, data, transform=None, tokenizer=None, cache=False):
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer

        with g_pathmgr.open(data, "rb") as f:
            self.samples = pickle.load(f)

        # init_cache()
        self.cache = cache

    def __getitem__(self, i):
        index, title, desc = self.samples[i]
        caption = np.random.choice([title, desc])
        img = yfcc_loader(self.root, index, cache=self.cache)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return Sample(data=(img, caption))

    def __len__(self):
        return len(self.samples)


class YFCCDatasetSLIP(torch.utils.data.Dataset):
    """
    YFCC dataset for SLIP:
    one view of an image with caption, two views of an uncaptioned image
    """

    def __init__(self, root, data, transform, augment, tokenizer=None):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer
        with g_pathmgr.open(data, "rb") as f:
            self.samples = pickle.load(f)

    def __getitem__(self, i):
        index, title, desc = self.samples[i]
        caption = np.random.choice([title, desc])
        img = yfcc_loader(self.root, index)

        img = self.transform(img)
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return Sample(data=(img, caption, aug1, aug2))

    def __len__(self):
        return len(self.samples)


class YFCCDatasetSSL(torch.utils.data.Dataset):
    """
    YFCC dataset for SSL:
    two views of an uncaptioned image
    """

    def __init__(self, root, data, augment):
        self.root = root
        self.augment = augment
        with g_pathmgr.open(data, "rb") as f:
            self.samples = pickle.load(f)

    def __getitem__(self, i):
        index, _, _ = self.samples[i]
        img = yfcc_loader(self.root, index)

        # apply transformation
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return Sample(data=(aug1, aug2))

    def __len__(self):
        return len(self.samples)


class CMD224Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="/large_experiments/cmd/images_224/",
        data="/large_experiments/cmd/cmd_complete_resized_224.npy",
        transform=None,
        tokenizer=None,
    ):
        self.transform = transform or (lambda x: x)
        self.tokenizer = tokenizer

        self.data_root = root
        self.txt = np.load(data, allow_pickle=True)
        self.txt = [
            {"image_name": d["image_name"], "text": d["captions"][0]} for d in self.txt
        ]

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, i):
        image_name = self.txt[i]["image_name"]
        try:
            image = Image.open(self.data_root + image_name)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except:
            print("Failed to load hte image at", i, image_name)

        text = self.txt[i]["text"]

        return Sample(data=(self.transform(image), self.tokenizer(text)))


class CMD224DatasetChunked(torch.utils.data.Dataset):
    def __init__(
        self,
        root="/large_experiments/cmd/images_224/",
        data="/checkpoint/kalyanv/data/cmd/cmd_meta_data_55m_chunked_1000",
        chunk_size=1000,
        transform=None,
        tokenizer=None,
    ):
        self.transform = transform
        self.tokenizer = tokenizer

        self.data_root = root
        self.meta_data_root = data
        self.chunk_size = chunk_size

    def __len__(self):
        return 55695156

    def __getitem__(self, i):
        in_chunk_idx = i % self.chunk_size
        chunk_id = i - in_chunk_idx

        with g_pathmgr.open(
            os.path.join(self.meta_data_root, f"chunk_{chunk_id}.pkl"), "rb"
        ) as f:
            sample = pickle.load(f)
            sample = sample[in_chunk_idx]

        # with g_pathmgr.open(os.path.join(self.data_root, sample["image_name"]), "rb") as fopen:
        #     image =  Image.open(fopen).convert("RGB")

        image = Image.open(os.path.join(self.data_root, sample["image_name"]))
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = sample["captions"][0]

        return Sample(data=(self.transform(image), self.tokenizer(text)))
