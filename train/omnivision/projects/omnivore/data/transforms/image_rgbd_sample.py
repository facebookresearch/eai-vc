from typing import Callable

import numpy as np
import torch
from omnivision.utils.generic import dataclass_as_dict
from omnivore.data.api import (
    BatchDepthSample,
    BatchDepthTextSample,
    BatchSample,
    BatchVisionDepthSample,
    BatchVisionDepthTextSample,
    BatchVisionSample,
    BatchVisionTextSample,
    Sample,
    VisionDepthSample,
    VisionDepthTextSample,
    VisionSample,
    VisionTextSample,
)
from omnivore.data.transforms.transform_wrappers import SingleFieldTransform

_SUPPORTED_VISION_ONLY_BATCHES = (BatchVisionSample, BatchVisionTextSample)
_SUPPORTED_VISION_DEPTH_BATCHES = (BatchVisionDepthSample, BatchVisionDepthTextSample)

_SUPPORTED_VISION_ONLY_SAMPLES = (VisionSample, VisionTextSample)
_SUPPORTED_VISION_DEPTH_SAMPLES = (VisionDepthSample, VisionDepthTextSample)


def create_vision_depth_concat_sample(sample):
    obj = dataclass_as_dict(sample)
    assert obj["vision"].shape[0] == 3
    assert obj["depth"].shape[0] == 1
    obj["vision"] = torch.cat([obj["vision"], obj["depth"]], dim=0)
    del obj["depth"]
    if "text" in obj:
        return VisionTextSample(**obj)
    else:
        return VisionSample(**obj)


def create_vision_depth_sample_from_concat(sample):
    obj = dataclass_as_dict(sample)
    assert obj["vision"].shape[0] == 4
    obj["depth"] = obj["vision"][3:, ...]
    obj["vision"] = obj["vision"][:3, ...]
    if "text" in obj:
        return VisionDepthTextSample(**obj)
    else:
        return VisionDepthSample(**obj)


def create_batch_vision_depth_sample_from_concat(batch):
    obj = dataclass_as_dict(batch)
    assert obj["vision"].ndim == 4
    assert obj["vision"].shape[1] == 4
    obj["depth"] = obj["vision"][:, 3:, ...]
    obj["vision"] = obj["vision"][:, :3, ...]
    if "text" in obj:
        return BatchVisionDepthTextSample(**obj)
    else:
        return BatchVisionDepthSample(**obj)


def create_batch_vision_only_sample(batch):
    obj = dataclass_as_dict(batch)
    assert "vision" in obj
    del obj["depth"]
    if "text" in obj:
        return BatchVisionTextSample(**obj)
    else:
        return BatchVisionSample(**obj)


def create_batch_depth_only_sample(batch):
    obj = dataclass_as_dict(batch)
    assert "depth" in obj
    del obj["vision"]
    if "text" in obj:
        return BatchDepthTextSample(**obj)
    else:
        return BatchDepthSample(**obj)


class VisionDepthConcatChannelTransform(SingleFieldTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("vision", *args, **kwargs)

    def __call__(self, sample: Sample) -> Sample:
        assert isinstance(
            sample, _SUPPORTED_VISION_DEPTH_SAMPLES
        ), f"Found sample of type {type(sample)}. Supported types {_SUPPORTED_VISION_DEPTH_SAMPLES}"
        sample = create_vision_depth_concat_sample(sample)
        return super().__call__(sample)


class VisionDepthConcatChannelToVisionDepth(Callable):
    def __call__(self, sample: Sample) -> Sample:
        assert isinstance(
            sample, _SUPPORTED_VISION_ONLY_SAMPLES
        ), f"Found sample of type {type(sample)}. Supported types {_SUPPORTED_VISION_ONLY_SAMPLES}"
        sample = create_vision_depth_sample_from_concat(sample)
        return sample


class VisionDepthConcatChannelToVisionDepthBatch(Callable):
    def __call__(self, batch: BatchVisionSample) -> BatchVisionDepthSample:
        assert isinstance(
            batch, _SUPPORTED_VISION_ONLY_BATCHES
        ), f"Found batch of type {type(batch)}. Supported types {_SUPPORTED_VISION_ONLY_BATCHES}"
        batch = create_batch_vision_depth_sample_from_concat(batch)
        return batch


class DropVisionDepth(Callable):
    """
    Batch-level transform that randomly drops either the .vision or the .depth field or neither in a batch
    Returns the appropriate type of BatchVisionSample or BatchDetphSample
    """

    def __init__(self, vision_depth_drop_prob: float) -> None:
        super().__init__()
        assert (
            vision_depth_drop_prob >= 0 and vision_depth_drop_prob <= 1
        ), f"Invalid value for vision_depth_drop_prob: {vision_depth_drop_prob}"
        self.vision_depth_drop_prob = vision_depth_drop_prob

    def __call__(self, batch) -> BatchSample:
        assert isinstance(batch, _SUPPORTED_VISION_DEPTH_BATCHES)

        to_drop = np.random.random() < self.vision_depth_drop_prob
        if to_drop:
            # select vision or depth fields randomly
            # do not drop both!
            drop_vision = np.random.random() < 0.5
            if drop_vision:
                batch = create_batch_vision_only_sample(batch)
            else:
                batch = create_batch_depth_only_sample(batch)
        return batch


class _DropDepthOrVision(Callable):
    def __init__(self, drop_prob: float, batch_create_fn: Callable) -> None:
        super().__init__()
        assert drop_prob >= 0 and drop_prob <= 1
        self.drop_prob = drop_prob
        self.batch_create_fn = batch_create_fn

    def __call__(self, batch, batch_create_fn=None) -> BatchSample:
        assert isinstance(
            batch, _SUPPORTED_VISION_DEPTH_BATCHES
        ), f"Found batch of type {type(batch)}. Supported types {_SUPPORTED_VISION_DEPTH_BATCHES}"
        to_drop = np.random.random() < self.drop_prob
        if to_drop:
            batch_create_fn = (
                batch_create_fn if batch_create_fn is None else self.batch_create_fn
            )
            batch = self.batch_create_fn(batch)
        return batch


class DropDepth(_DropDepthOrVision):
    def __init__(self, depth_drop_prob: float) -> None:
        super().__init__(
            depth_drop_prob, batch_create_fn=create_batch_vision_only_sample
        )


class DropVision(_DropDepthOrVision):
    def __init__(self, vision_drop_prob: float) -> None:
        super().__init__(
            vision_drop_prob, batch_create_fn=create_batch_depth_only_sample
        )
