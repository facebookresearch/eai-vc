from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch

from omnivision.utils.generic import (
    convert_int_or_intlist_to_one_or_multi_hot,
    dataclass_as_dict,
)

from torch.utils.data.dataloader import default_collate


@dataclass
class Batch:
    # the following are per batch args which are passed to the trainer
    # and are set to reasonable defaults
    model_fwd_kwargs: Dict = field(default_factory=dict)
    accum_steps: int = 1


def create_batch_sample_cls(cls):
    """Dynamically creates a dataclass which is a `Batch` and a `Sample`.

    This function also registers the class in globals() to make the class picklable.
    """
    cls_name = f"{Batch.__name__}{cls.__name__}"
    batch_sample_cls = make_dataclass(cls_name, fields=(), bases=(cls, Batch))
    batch_sample_cls.__module__ = __name__
    globals()[cls_name] = batch_sample_cls
    return cls


@create_batch_sample_cls
@dataclass
class Sample:
    # NOTE: Up to Python 3.9, dataclasses don't support inheritance when there
    # are both positional and default arguments present. See
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    data_idx: int = None
    data_valid: bool = None
    label: Any = None

    @classmethod
    def get_batch_sample_class(cls):
        return globals()[f"{Batch.__name__}{cls.__name__}"]


@create_batch_sample_cls
@dataclass
class DepthSample(Sample):
    # depth "D" channel image
    depth: Any = None


@create_batch_sample_cls
@dataclass
class TextSample(Sample):
    # Corresponds to the data, a list of strings typically
    text: Any = None


@create_batch_sample_cls
@dataclass
class VisionSample(Sample):
    vision: Any = None


@create_batch_sample_cls
@dataclass
class VisionMaskSample(VisionSample):
    mask: Any = None


@create_batch_sample_cls
@dataclass
class VisionTextSample(VisionSample, TextSample):
    pass


@create_batch_sample_cls
@dataclass
class VisionTextHashtagSample(VisionSample):
    text: Any = None
    hashtags: Any = None


@create_batch_sample_cls
@dataclass
class VisionDepthSample(VisionSample, DepthSample):
    pass


@create_batch_sample_cls
@dataclass
class VisionDepthTextSample(VisionSample, DepthSample, TextSample):
    pass


@create_batch_sample_cls
@dataclass
class DepthTextSample(DepthSample, TextSample):
    pass


@create_batch_sample_cls
@dataclass
class AudioSample(Sample):
    audio: Any = None


@create_batch_sample_cls
@dataclass
class VisionAudioSample(VisionSample, AudioSample):
    pass


@create_batch_sample_cls
@dataclass
class AudioTextSample(AudioSample, TextSample):
    pass


UPGRADE_TO_TEXT_SAMPLE = {
    VisionSample: VisionTextSample,
    VisionDepthSample: VisionDepthTextSample,
    DepthSample: DepthTextSample,
    AudioSample: AudioTextSample,
}

# Fields that contain data (not labels, etc). Basically stuff that might
# require a network forward.
# These firlds can *potentially* contain list of tensors (instead of tensors).
# This information is used to do, for instance,
# list inferece when handle_list_inputs=True. These fields are expected
# to have lists in them in that case.
DATA_FIELDS = {
    "text",
    "vision",
    "audio",
    "depth",
    "mask",
}


class DefaultOmnivoreCollator(Callable):
    def __init__(
        self,
        output_key: str,
        batch_kwargs=None,
        batch_transforms=None,
        input_batch_is_collated=False,
        # Set this to the number of classes to convert the label
        # to 1-hot. Espl useful for multilabel datasets.
        convert_label_to_one_hot_num_classes=-1,
    ) -> None:
        self.output_key = output_key
        self.batch_kwargs = batch_kwargs
        self.batch_transforms = batch_transforms
        self.input_batch_is_collated = input_batch_is_collated
        self.convert_label_to_one_hot_num_classes = convert_label_to_one_hot_num_classes

    def process_sample_to_dict(self, sample):
        res = dataclass_as_dict(sample)
        if self.convert_label_to_one_hot_num_classes > 0:
            res["label"] = convert_int_or_intlist_to_one_or_multi_hot(
                res["label"], self.convert_label_to_one_hot_num_classes
            )
        return res

    def collate_batch(self, batch_in):
        batch = []
        assert len(batch_in) > 0
        for sample in batch_in:
            assert isinstance(sample, Sample), f"Found {type(sample)}"
            batch.append(self.process_sample_to_dict(sample))
        return batch, type(batch_in[0])

    def __call__(self, batch_in):
        if self.input_batch_is_collated:
            batch = batch_in
        else:
            batch, sample_cls = self.collate_batch(batch_in)
            batch_cls = sample_cls.get_batch_sample_class()
            batch = batch_cls(**default_collate(batch))
        if self.batch_kwargs is not None:
            batch_field_names = {f.name for f in fields(Batch)}
            for key, value in self.batch_kwargs.items():
                assert key in batch_field_names
                setattr(batch, key, value)

        if self.batch_transforms is not None:
            for transform in self.batch_transforms:
                batch = transform(batch)

        if self.output_key is not None:
            batch = {self.output_key: batch}
        return batch


class SampleListOmnivoreCollator(DefaultOmnivoreCollator):
    def collate_batch(self, batch_in):
        """
        In this case each batch element is a list of Samples.
        This happens, for eg, when using replicate for MAE training where the same
        sample is replicated N times and augmented those many times. Here we collate
        the list into a single list.
        """
        batch = []
        assert len(batch_in) > 0
        for samples in batch_in:
            assert isinstance(samples, list), f"Found {type(samples)}"
            assert all(
                [isinstance(el, Sample) for el in samples]
            ), f"Found {[type(el) for el in samples]}"
            batch += [self.process_sample_to_dict(el) for el in samples]
        sample_cls = type(batch_in[0][0])
        return batch, sample_cls
