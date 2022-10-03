import io
import shutil
import sys
import tempfile
import time
from copy import deepcopy

import numpy as np
import pytorch_lightning
import torch
import tqdm
from iopath.common.file_io import g_pathmgr

from omnivore.data.path_dataset import ImagePathDataset, VideoPathDataset
from omnivore.fb.data.async_path_dataset import (
    AsyncImagePathDataset,
    AsyncVideoPathDataset,
)
from PIL import Image


def empty_collator(*args, **kwargs):
    return None


class TestImageDatasetCreate:
    def __init__(
        self,
        length=10_000,
        path="manifold://omnivore/tree/datasets/imagenet_full_size/train/n15075141/n15075141_9933.JPEG",
    ):
        self.out_dir = tempfile.mkdtemp()
        file_name = path.split("/")[-1]
        self.local_path = f"{self.out_dir}/{file_name}"
        g_pathmgr.copy(path, self.local_path)
        paths = np.array([self.local_path] * length)
        label = 0
        labels = np.array([label] * length)
        self.path_file = f"{self.out_dir}/paths.npy"
        self.label_file = f"{self.out_dir}/labels.npy"
        np.save(self.path_file, paths)
        np.save(self.label_file, labels)
        print_msg(f"Created test directory: {self.out_dir}")

    def get_file_path(self):
        return self.local_path

    def get_paths(self):
        return self.path_file, self.label_file

    def cleanup(self):
        shutil.rmtree(self.out_dir)
        print_msg(f"Removed test directory: {self.out_dir}")


class TestVideoDatasetCreate(TestImageDatasetCreate):
    def __init__(
        self,
        length=10_000,
        path="manifold://omnivore/tree/dataset/kinetics400_high_qual_320_trimmed/train/air_drumming/cXY4ENE36Jo.mp4",
    ):
        super().__init__(length, path)


class TestAsyncVideoPathDataset(AsyncVideoPathDataset):
    def __init__(self, *args, skip_decode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_decode = skip_decode

    async def __getitem__(self, idx):
        path = self._get_path(idx)
        if self.skip_decode:
            return await g_pathmgr.opena(path, "rb").read()
        return await super()._get_video_object(path)


class TestAsyncImagePathDataset(AsyncImagePathDataset):
    def __init__(self, *args, skip_decode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_decode = skip_decode

    async def __getitem__(self, idx):
        path = self._get_path(idx)
        if self.skip_decode:
            return await g_pathmgr.opena(path, "rb").read()
        return await super().load_object(path)


class TestVideoPathDataset(VideoPathDataset):
    def __init__(self, *args, skip_decode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_decode = skip_decode

    def __getitem__(self, idx):
        path = self._get_path(idx)
        if self.skip_decode:
            with g_pathmgr.open(path, "rb") as fh:
                return io.BytesIO(fh.read())
        return super()._get_video_object(path)


class TestImagePathDataset(ImagePathDataset):
    def __init__(self, *args, skip_decode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_decode = skip_decode

    def __getitem__(self, idx):
        path = self._get_path(idx)
        if self.skip_decode:
            with g_pathmgr.open(path, "rb") as fh:
                return io.BytesIO(fh.read())
        return super().load_object(path)


def get_image_transforms():
    import omnivore.data.transforms.rand_auto_aug
    import torchvision.transforms

    return torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.RandomResizedCrop(size=224, interpolation=3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            omnivore.data.transforms.rand_auto_aug.RandAugment(
                magnitude=9,
                magnitude_std=0.5,
                increasing_severity=True,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.25),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def get_image_dataset_args(
    use_memcache=False,
    use_transforms=True,
    path_file="manifold://omnivore/tree/datasets/imagenet_1k_meta/train_images_manifold_v2.npy",
    label_file="manifold://omnivore/tree/datasets/imagenet_1k_meta/train_labels.npy",
    remove_prefix=None,
    new_prefix=None,
):
    import torchvision.transforms

    assert (remove_prefix is None) == (new_prefix is None)

    if remove_prefix is None:
        remove_prefix = "manifold://"
    if new_prefix is None:
        new_prefix = "memcache_manifold://" if use_memcache else "manifold://"
    else:
        if use_memcache:
            raise ValueError("Cannot set memcache with a non None new_prefix")

    return dict(  # noqa
        path_file_list=[path_file],
        label_file_list=[label_file],
        remove_prefix=remove_prefix,
        new_prefix=new_prefix,
        transform=get_image_transforms()
        if use_transforms
        else torchvision.transforms.Compose([]),
    )


def get_video_clip_sampler():
    import pytorchvideo.data.clip_sampling

    return pytorchvideo.data.clip_sampling.RandomClipSampler(clip_duration=2)


def get_video_frame_sampler():
    import pytorchvideo.transforms

    return pytorchvideo.transforms.UniformTemporalSubsample(num_samples=32)


def get_video_transforms():
    import pytorchvideo.transforms
    import torchvision.transforms
    import torchvision.transforms._transforms_video

    return torchvision.transforms.Compose(
        transforms=[
            pytorchvideo.transforms.ShortSideScale(size=256),
            torchvision.transforms.RandomResizedCrop(size=224, interpolation=3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms._transforms_video.NormalizeVideo(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def get_video_dataset_args(
    use_memcache=False,
    use_transforms=True,
    decoder=None,
    decoder_kwargs=None,
    path_file="manifold://omnivore/tree/datasets/kinetics_400_meta/vidpaths_train.npy",
    label_file="manifold://omnivore/tree/datasets/kinetics_400_meta/labels_train.npy",
    remove_prefix=None,
    new_prefix=None,
):
    import torchvision.transforms

    assert (remove_prefix is None) == (new_prefix is None)

    if remove_prefix is None:
        remove_prefix = "manifold://"
    if new_prefix is None:
        new_prefix = "memcache_manifold://" if use_memcache else "manifold://"
    else:
        if use_memcache:
            raise ValueError("Cannot set memcache with a non None new_prefix")

    return dict(  # noqa
        path_file_list=[path_file],
        label_file_list=[label_file],
        remove_prefix=remove_prefix,
        new_prefix=new_prefix,
        clip_sampler=get_video_clip_sampler(),
        frame_sampler=get_video_frame_sampler(),
        decoder="pyav" if decoder is None else decoder,
        decoder_kwargs=decoder_kwargs,
        normalize_to_0_1=False,
        transform=get_video_transforms()
        if use_transforms
        else torchvision.transforms.Compose([]),
    )


def time_dataset(
    msg,
    dataset,
    batch_size,
    num_workers,
    collate_fn,
    shape=None,
    skip_batches=10,
    measure_batches=100,
):
    print_msg(msg)
    # dataloader = iter(dataset)
    mp_ctx = (
        torch.multiprocessing.get_context(method="spawn") if num_workers > 0 else None
    )
    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_size=batch_size,
            multiprocessing_context=mp_ctx,
        )
    )
    start = None
    end = None
    for i, sample in enumerate(tqdm.tqdm(dataloader, file=sys.stdout)):
        if shape is not None:
            expected_shape = (batch_size,) + shape
            assert (
                sample.data.shape == expected_shape
            ), f"{sample.data.shape} doesn't match {expected_shape}"
        if i == skip_batches:
            start = time.perf_counter()
        if i == skip_batches + measure_batches - 1:
            end = time.perf_counter()
            break

    if end is None:
        raise ValueError("Not enough batches")
    speed = measure_batches / (end - start)
    print_msg(
        f"{msg}. Speed: {speed} batches/sec, batch_size: {batch_size}, num_workers: {num_workers}, skip_batches: {skip_batches}, measure_batches: {measure_batches}"
    )


def time_op(msg, op, *op_args, skip_iters=10, measure_iters=100, **op_kwargs):
    for i in tqdm.tqdm(range(measure_iters + skip_iters)):
        op(*op_args, **op_kwargs)
        if i == skip_iters:
            start = time.perf_counter()
    assert i == skip_iters + measure_iters - 1
    end = time.perf_counter()
    speed = measure_iters / (end - start)
    print_msg(
        f"{msg}. Speed: {speed} iters/sec, skip_iters: {skip_iters}, measure_iters: {measure_iters}"
    )


def print_msg(msg):
    pattern = "-" * 80
    print(pattern, msg, pattern, sep="\n")


def main():
    trainer = pytorch_lightning.Trainer(
        num_nodes=1,
        gpus=1,
        replace_sampler_ddp=False,
        max_epochs=1,
        accelerator="cpu",
        strategy="ddp",
    )
    print_msg("Initializing torch.distributed")
    trainer.strategy.setup_environment()

    from omnivision.trainer.distributed import init_ranks

    init_ranks(trainer)

    import omnivore.data.api
    import omnivore.data.path_dataset
    import omnivore.fb.data.async_path_dataset

    num_workers = 3
    # IN1k async sync
    if False:
        for memcache in [False, True]:
            for transforms in [False, True]:
                time_dataset(
                    f"Async image memcache={memcache}, transforms={transforms}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=250,
                        dataset=omnivore.fb.data.async_path_dataset.AsyncImagePathDataset(
                            **get_image_dataset_args(
                                use_memcache=memcache, use_transforms=transforms
                            )
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 224, 224) if transforms else None,
                )
                time_dataset(
                    f"Sync image memcache={memcache}, transforms={transforms}",
                    omnivore.data.path_dataset.ImagePathDataset(
                        **get_image_dataset_args(
                            use_memcache=memcache, use_transforms=transforms
                        )
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 224, 224) if transforms else None,
                )
    # K400 async sync
    if False:
        for memcache in [False, True]:
            for transforms in [False, True]:
                time_dataset(
                    f"Async video memcache={memcache}, transforms={transforms}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=250,
                        dataset=omnivore.fb.data.async_path_dataset.AsyncVideoPathDataset(
                            **get_video_dataset_args(
                                use_memcache=memcache, use_transforms=transforms
                            )
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
                time_dataset(
                    f"Sync video memcache={memcache}, transforms={transforms}",
                    omnivore.data.path_dataset.VideoPathDataset(
                        **get_video_dataset_args(
                            use_memcache=memcache, use_transforms=transforms
                        )
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
    # K400 async sync decord
    if False:
        decoder = "decord"
        for memcache in [False, True]:
            for transforms in [False, True]:
                time_dataset(
                    f"Async video memcache={memcache}, transforms={transforms}, decoder={decoder}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=250,
                        dataset=omnivore.fb.data.async_path_dataset.AsyncVideoPathDataset(
                            **get_video_dataset_args(
                                use_memcache=memcache,
                                use_transforms=transforms,
                                decoder=decoder,
                            )
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
                time_dataset(
                    f"Sync video memcache={memcache}, transforms={transforms}, decoder={decoder}",
                    omnivore.data.path_dataset.VideoPathDataset(
                        **get_video_dataset_args(
                            use_memcache=memcache,
                            use_transforms=transforms,
                            decoder=decoder,
                        )
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
    # K400 skip clip sampler / decoding
    if False:
        for memcache in [False, True]:
            for skip_decode in [False, True]:
                time_dataset(
                    f"Async no sampler video memcache={memcache}, skip_decode={skip_decode}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=250,
                        dataset=TestAsyncVideoPathDataset(
                            **get_video_dataset_args(
                                use_memcache=memcache, use_transforms=False
                            ),
                            skip_decode=skip_decode,
                        ),
                    ),
                    collate_fn=empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=None,
                    skip_batches=2,
                    measure_batches=10,
                )
                time_dataset(
                    f"Sync no sampler video memcache={memcache}, skip_decode={skip_decode}",
                    TestVideoPathDataset(
                        **get_video_dataset_args(
                            use_memcache=memcache, use_transforms=False
                        ),
                        skip_decode=skip_decode,
                    ),
                    collate_fn=empty_collator,
                    batch_size=32,
                    num_workers=num_workers,
                    shape=None,
                    skip_batches=2,
                    measure_batches=10,
                )
    # IN1k sync local test dataset
    if False:
        test_dataset = TestImageDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        memcache = False
        for workers in range(8):
            for transforms in [False, True]:
                time_dataset(
                    f"Sync image memcache={memcache}, workers={workers}, transforms={transforms}",
                    omnivore.data.path_dataset.ImagePathDataset(
                        **get_image_dataset_args(
                            path_file=path_file,
                            label_file=label_file,
                            use_memcache=memcache,
                            use_transforms=transforms,
                            remove_prefix="",
                            new_prefix="",
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=(3, 224, 224) if transforms else None,
                    skip_batches=10,
                    measure_batches=100,
                )
        test_dataset.cleanup()
    # K400 sync local test dataset decord threads
    if False:
        test_dataset = TestVideoDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        decoder = "decord"
        memcache = False
        threads = 0
        for workers in range(8):
            # for threads in [1, 2, 3, 4, 0]:  # 0 == auto
            for transforms in [False, True]:
                time_dataset(
                    f"Sync video memcache={memcache}, workers={workers}, transforms={transforms}, decoder={decoder}, threads={threads}",
                    omnivore.data.path_dataset.VideoPathDataset(
                        **get_video_dataset_args(
                            path_file=path_file,
                            label_file=label_file,
                            use_memcache=memcache,
                            use_transforms=transforms,
                            decoder=decoder,
                            decoder_kwargs={"num_threads": threads},
                            remove_prefix="",
                            new_prefix="",
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
        test_dataset.cleanup()
    # K400 sync local test dataset pyav
    if False:
        test_dataset = TestVideoDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        decoder = "pyav"
        memcache = False
        for workers in range(8):
            for transforms in [False, True]:
                time_dataset(
                    f"Sync video memcache={memcache}, workers={workers}, transforms={transforms}, decoder={decoder}",
                    omnivore.data.path_dataset.VideoPathDataset(
                        **get_video_dataset_args(
                            path_file=path_file,
                            label_file=label_file,
                            use_memcache=memcache,
                            use_transforms=transforms,
                            decoder=decoder,
                            remove_prefix="",
                            new_prefix="",
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
        test_dataset.cleanup()
    # K400 sync local test dataset skip clip sampler
    if False:
        test_dataset = TestVideoDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        decoder = "pyav"
        memcache = False
        transforms = False
        for workers in range(8):
            for skip_decode in [False, True]:
                time_dataset(
                    f"Sync no sampler video memcache={memcache}, skip_decode={skip_decode}",
                    TestVideoPathDataset(
                        **get_video_dataset_args(
                            path_file=path_file,
                            label_file=label_file,
                            use_memcache=memcache,
                            use_transforms=transforms,
                            decoder=decoder,
                            remove_prefix="",
                            new_prefix="",
                        ),
                        skip_decode=skip_decode,
                    ),
                    collate_fn=empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=None,
                    skip_batches=2,
                    measure_batches=10,
                )
        test_dataset.cleanup()
    # IN1k sync local test dataset skip decode
    if False:
        test_dataset = TestImageDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        memcache = False
        transforms = False
        for workers in range(8):
            for skip_decode in [False, True]:
                time_dataset(
                    f"Sync image memcache={memcache}, skip_decode={skip_decode}",
                    TestImagePathDataset(
                        **get_image_dataset_args(
                            path_file=path_file,
                            label_file=label_file,
                            use_memcache=memcache,
                            use_transforms=transforms,
                            remove_prefix="",
                            new_prefix="",
                        ),
                        skip_decode=skip_decode,
                    ),
                    collate_fn=empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=None,
                    skip_batches=10,
                    measure_batches=100,
                )
        test_dataset.cleanup()
    # K400 async decord prefetch length
    if False:
        decoder = "decord"
        memcache = False
        transforms = True
        # for workers in range(4):
        for workers in [4, 5, 6, 7]:
            for max_prefetch in [2, 8, 32, 128, 256, 512, 1024]:
                time_dataset(
                    f"Async video memcache={memcache}, transforms={transforms}, decoder={decoder}, max_prefetch={max_prefetch}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=max_prefetch,
                        dataset=omnivore.fb.data.async_path_dataset.AsyncVideoPathDataset(
                            **get_video_dataset_args(
                                use_memcache=memcache,
                                use_transforms=transforms,
                                decoder=decoder,
                            )
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=(3, 32, 224, 224) if transforms else None,
                    skip_batches=2,
                    measure_batches=10,
                )
    # IN1k async prefetch length
    if False:
        memcache = False
        transforms = True
        for workers in [4, 5, 6, 7]:
            # for workers in range(4):
            for max_prefetch in [2, 8, 32, 128, 512]:
                time_dataset(
                    f"Async image memcache={memcache}, transforms={transforms}, max_prefetch={max_prefetch}",
                    omnivore.fb.data.async_path_dataset.AsyncToIterableDataset(
                        max_prefetch=max_prefetch,
                        dataset=omnivore.fb.data.async_path_dataset.AsyncImagePathDataset(
                            **get_image_dataset_args(
                                use_memcache=memcache, use_transforms=transforms
                            )
                        ),
                    ),
                    collate_fn=omnivore.data.api.DefaultOmnivoreCollator()
                    if transforms
                    else empty_collator,
                    batch_size=32,
                    num_workers=workers,
                    shape=(3, 224, 224) if transforms else None,
                )
    # IN1k sync local test ops
    if False:
        test_dataset = TestImageDatasetCreate()
        file_path = test_dataset.get_file_path()
        path_file, label_file = test_dataset.get_paths()
        time_op(
            "Read file", read_file, path=file_path, skip_iters=100, measure_iters=1000
        )
        data = read_file(file_path)
        time_op(
            "Data to stream",
            data_to_stream,
            vision=data,
            skip_iters=100,
            measure_iters=1000,
        )
        stream = data_to_stream(data)
        time_op(
            "Open image from stream",
            open_image_from_stream,
            stream=stream,
            skip_iters=100,
            measure_iters=1000,
        )
        image = open_image_from_stream(stream)
        time_op(
            "Apply transforms",
            apply_image_transforms,
            image=image,
            transforms=get_image_transforms(),
            skip_iters=100,
            measure_iters=1000,
        )
        time_op(
            "Get item",
            get_item_image,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            skip_iters=100,
            measure_iters=1000,
        )
        test_dataset.cleanup()
    # K400 sync local test ops
    if False:
        test_dataset = TestVideoDatasetCreate()
        file_path = test_dataset.get_file_path()
        path_file, label_file = test_dataset.get_paths()
        time_op(
            "Read file", read_file, path=file_path, skip_iters=100, measure_iters=1000
        )
        data = read_file(file_path)
        time_op(
            "Data to stream",
            data_to_stream,
            vision=data,
            skip_iters=100,
            measure_iters=1000,
        )
        stream = data_to_stream(data)
        time_op(
            "Open video + audio from stream - pyav",
            open_video_from_stream,
            stream=stream,
            decoder="pyav",
            decode_audio=True,
            skip_iters=100,
            measure_iters=1000,
        )
        time_op(
            "Open video - audio from stream - pyav",
            open_video_from_stream,
            stream=stream,
            decoder="pyav",
            decode_audio=False,
            skip_iters=100,
            measure_iters=1000,
        )
        time_op(
            "Open video + audio from stream - decord",
            open_video_from_stream,
            stream=stream,
            decoder="decord",
            decode_audio=True,
            skip_iters=100,
            measure_iters=1000,
        )
        time_op(
            "Open video - audio from stream - decord",
            open_video_from_stream,
            stream=stream,
            decoder="decord",
            decode_audio=False,
            skip_iters=100,
            measure_iters=1000,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="pyav", decode_audio=True)
        time_op(
            "Get clip + audio - pyav",
            get_clip,
            video=video,
            clip_sampler=get_video_clip_sampler(),
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="pyav", decode_audio=False)
        time_op(
            "Get clip - audio - pyav",
            get_clip,
            video=video,
            clip_sampler=get_video_clip_sampler(),
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="decord", decode_audio=True)
        time_op(
            "Get clip + audio - decord",
            get_clip,
            video=video,
            clip_sampler=get_video_clip_sampler(),
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="decord", decode_audio=False)
        time_op(
            "Get clip - audio - decord",
            get_clip,
            video=video,
            clip_sampler=get_video_clip_sampler(),
            skip_iters=2,
            measure_iters=20,
        )
        clip = get_clip(video, get_video_clip_sampler())
        time_op(
            "Get frames",
            get_frames,
            clip=clip,
            frame_sampler=get_video_frame_sampler(),
            skip_iters=10,
            measure_iters=100,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="pyav", decode_audio=True)
        time_op(
            "Load video object + audio clip - pyav",
            load_video_object,
            video=video,
            dataset=omnivore.data.path_dataset.VideoPathDataset(
                **get_video_dataset_args()
            ),
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="pyav", decode_audio=False)
        time_op(
            "Load video object - audio clip - pyav",
            load_video_object,
            video=video,
            dataset=omnivore.data.path_dataset.VideoPathDataset(
                **get_video_dataset_args()
            ),
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item + audio clip - pyav",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="pyav",
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item + audio clip - decord",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="decord",
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="decord", decode_audio=True)
        time_op(
            "Load video object + audio clip - decord",
            load_video_object,
            video=video,
            dataset=omnivore.data.path_dataset.VideoPathDataset(
                **get_video_dataset_args()
            ),
            skip_iters=2,
            measure_iters=20,
        )
        stream = deepcopy(stream)
        video = open_video_from_stream(stream, decoder="decord", decode_audio=False)
        time_op(
            "Load video object - audio clip - decord",
            load_video_object,
            video=video,
            dataset=omnivore.data.path_dataset.VideoPathDataset(
                **get_video_dataset_args()
            ),
            skip_iters=2,
            measure_iters=20,
        )
        video_clip = load_video_object(
            video,
            dataset=omnivore.data.path_dataset.VideoPathDataset(
                **get_video_dataset_args()
            ),
        )
        time_op(
            "Apply transforms",
            apply_video_transforms,
            video=video_clip,
            transforms=get_video_transforms(),
            skip_iters=100,
            measure_iters=1000,
        )
        test_dataset.cleanup()
    # K400 sync local test read
    if True:
        test_dataset = TestVideoDatasetCreate()
        path_file, label_file = test_dataset.get_paths()
        time_op(
            "Get item - real torchvision",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="real_torchvision",
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item - real torchvision GPU",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="real_torchvision",
            decoder_kwargs={"device": "cuda"},
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item - pyav",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="pyav",
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item - decord",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="decord",
            skip_iters=2,
            measure_iters=20,
        )
        time_op(
            "Get item - torchvision",
            get_item_video,
            path_file=path_file,
            label_file=label_file,
            remove_prefix="",
            new_prefix="",
            decoder="torchvision",
            skip_iters=2,
            measure_iters=20,
        )
        test_dataset.cleanup()


def read_file(path):
    with g_pathmgr.open(path, "rb") as fopen:
        return fopen.read()


def data_to_stream(data):
    return io.BytesIO(data)


def open_image_from_stream(stream):
    return Image.open(stream).convert("RGB")


def apply_image_transforms(image, transforms):
    return transforms(image)


def open_video_from_stream(stream, decoder, decode_audio):
    from pytorchvideo.data.encoded_video import select_video_class

    stream = deepcopy(stream)
    video_cls = select_video_class(decoder)
    return video_cls(stream, decode_audio=decode_audio)


def load_video_object(video, dataset):
    def return_video(path):
        return video

    dataset._get_video_object = return_video
    return dataset.load_object(video)


def get_item_video(*args, **kwargs):
    import omnivore.data.path_dataset

    dataset = omnivore.data.path_dataset.VideoPathDataset(
        **get_video_dataset_args(*args, **kwargs)
    )

    return dataset[0]


def get_item_image(*args, **kwargs):
    import omnivore.data.path_dataset

    dataset = omnivore.data.path_dataset.ImagePathDataset(
        **get_image_dataset_args(*args, **kwargs)
    )

    return dataset[0]


def apply_video_transforms(video, transforms):
    return transforms(video)


def get_clip(video, clip_sampler):
    start, end, _, _, _ = clip_sampler(0, video.duration, annotation=None)
    return video.get_clip(start, end)["video"]


def get_frames(clip, frame_sampler):
    return frame_sampler(clip)


if __name__ == "__main__":
    main()
