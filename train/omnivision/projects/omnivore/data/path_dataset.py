# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
import torch

try:
    import torchaudio
except:
    logging.warn("torchaudio is not installed. Please install it via conda.")
import torchvision.transforms.functional as tvf
from iopath.common.file_io import g_pathmgr
from omnivision.utils.generic import dataclass_as_dict
from omnivore.data.api import (
    AudioSample,
    UPGRADE_TO_TEXT_SAMPLE,
    VisionAudioSample,
    VisionDepthSample,
    VisionSample,
)

from omnivore.data.real_torchvision_video_reader import RealTorchvisionVideoReader
from omnivore.utils.data import (
    FileLoader,
    flatten_list,
    get_local_dst_dir_to_copy,
    get_mean_image,
    IdentityTransform,
    SharedMemoryNumpyLoader,
    smart_copy_path,
)
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset


IDENTITY_TRANSFORM = IdentityTransform()
DEFAULT_SPATIAL_SIZE = 224
DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

# Label types
LABEL_TYPE_INT = "int"  # default, imagenet, kinetics type labels
# Comma separated string containing integer values (eg "10,20,30")
LABEL_TYPE_CSV = "csv"


class PathDataset(Dataset, ABC):
    def __init__(
        self,
        path_file_list: List[str],
        label_file_list: List[str],
        remove_prefix="",
        new_prefix="",
        remove_suffix="",
        new_suffix="",
        label_type: str = LABEL_TYPE_INT,
        copy_on_read: bool = False,
        copy_on_read_dst_basename: str = "data",
        transforms=None,
    ):
        """Creates a dataset where the metadata is stored in a numpy file.

        path_file_list: A list of paths which contain the path metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        label_file_list: A list of paths which contain the label metadata file. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the metadata.
        copy_on_read: bool variable indicating whether we copy over data
        copy_on_read_dst_basename: A string indicating where the basename of the directory we copy the `path` over to
        """
        self.is_initialized = False
        self.path_file_list = path_file_list
        self.label_file_list = label_file_list
        self.copy_on_read = copy_on_read
        self.transforms = [] if transforms is None else transforms

        self.remove_prefix = remove_prefix
        self.new_prefix = new_prefix
        self.remove_suffix = remove_suffix
        self.new_suffix = new_suffix
        self.label_type = label_type

        self.paths = None
        self.labels = None
        self.file_idx = None

        # used for shared memory
        self.label_sm_loader = SharedMemoryNumpyLoader()
        self.path_sm_loader = SharedMemoryNumpyLoader()

        self._load_data()
        self.num_samples = len(self.paths)
        assert len(self.paths) == len(
            self.labels
        ), f"Paths ({len(self.paths)}) != labels ({len(self.labels)})"
        logging.info(
            f"Created dataset from {self.path_file_list} of length: {self.num_samples}"
        )
        if self.copy_on_read:
            self.copy_on_read_dst_dir = get_local_dst_dir_to_copy(
                copy_on_read_dst_basename
            )
            logging.info(f"Will copy on read to {self.copy_on_read_dst_dir}")

    def _load_data(self):
        logging.info(f"Loading {self.label_file_list} with shared memory")
        self.labels, label_file_idx = self.label_sm_loader.load(self.label_file_list)
        logging.info(f"Loading {self.path_file_list} with shared memory")
        self.paths, path_file_idx = self.path_sm_loader.load(self.path_file_list)
        assert (
            label_file_idx == path_file_idx
        ), "Label file and path file were not found at the same index"
        self.is_initialized = True
        self.file_idx = path_file_idx

    def _replace_path_prefix(self, path, replace_prefix, new_prefix):
        if replace_prefix == "":
            path = new_prefix + path
        elif path.startswith(replace_prefix):
            # Replace might replace other instances of the prefix too, so only
            # replace the first one
            return new_prefix + path[len(replace_prefix) :]
        else:
            raise ValueError(f"Cannot replace `{replace_prefix}`` prefix in `{path}`")
        return path

    def _replace_path_suffix(self, path, replace_suffix, new_suffix):
        if replace_suffix == "":
            path = path + new_suffix
        elif path.endswith(replace_suffix):
            # Replace might replace other instances of the suffix too, so only
            # replace the last one
            return path[: -len(replace_suffix)] + new_suffix
        else:
            raise ValueError(f"Cannot replace `{replace_suffix}`` suffix in `{path}`")
        return path

    def __len__(self):
        return self.num_samples

    @abstractmethod
    def default_generator(self):
        pass

    @abstractmethod
    def load_object(self, path):
        pass

    def _get_path(self, idx):
        path = self._replace_path_prefix(
            self.paths[idx],
            replace_prefix=self.remove_prefix,
            new_prefix=self.new_prefix,
        )
        path = self._replace_path_suffix(
            path, replace_suffix=self.remove_suffix, new_suffix=self.new_suffix
        )
        if self.copy_on_read:
            path = smart_copy_path(path, dst_dir=self.copy_on_read_dst_dir)
        return path

    def try_load_object(self, idx):
        is_success = True
        try:
            data = self.load_object(self._get_path(idx))
        except Exception:
            logging.warning(f"Couldn't load: {self.paths[idx]}.")
            logging.debug("Exception: ", exc_info=True)
            is_success = False
            data = self.default_generator()
        return data, is_success

    def get_label(self, idx):
        return None if self.labels is None else self.labels[idx]

    @staticmethod
    def create_sample(idx, data, label, is_success):
        # Not casting the label to int since it can be anything, eg comma
        # sep string in case of multilabel datasets
        return VisionSample(
            vision=data, label=label, data_idx=idx, data_valid=is_success
        )

    def apply_transforms(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def _process_label(self, label: Union[int, str]) -> Union[int, List[int]]:
        if self.label_type == LABEL_TYPE_INT:
            assert isinstance(label, (int, np.integer))
            return label
        elif self.label_type == LABEL_TYPE_CSV:
            assert isinstance(label, str)
            return [int(el) for el in label.split(",")]
        else:
            raise NotImplementedError(f"Unkonwn label type {self.label_type}")

    def __getitem__(self, idx):
        data, is_success = self.try_load_object(idx)
        label = self.get_label(idx)
        sample = self.create_sample(idx, data, label, is_success)
        sample = self.apply_transforms(sample)
        sample.label = self._process_label(sample.label)
        return sample


class ImagePathDataset(PathDataset):
    def default_generator(self):
        return get_mean_image(DEFAULT_SPATIAL_SIZE)

    def load_object(self, path) -> Image.Image:
        with g_pathmgr.open(path, "rb") as fopen:
            return Image.open(fopen).convert("RGB")


class ImageWithDepthPathDataset(ImagePathDataset):
    def __init__(
        self,
        depth_path_file_list: List[str],
        concatenate_depth_and_rgb_channels: bool = True,
        load_image_as_tensor: bool = True,
        *args,
        remove_depth_prefix="",
        new_depth_prefix="",
        remove_depth_suffix="",
        new_depth_suffix="",
        **kwargs,
    ):
        """
        Shared Memory dataloader for RGB+Depth datasets.
        """
        super().__init__(*args, **kwargs)

        self.depth_path_file_list = depth_path_file_list
        self.concatenate_depth_and_rgb_channels = concatenate_depth_and_rgb_channels
        self.load_image_as_tensor = load_image_as_tensor

        self.remove_depth_prefix = remove_depth_prefix
        self.new_depth_prefix = new_depth_prefix
        self.remove_depth_suffix = remove_depth_suffix
        self.new_depth_suffix = new_depth_suffix

        self.depth_path_sm_loader = SharedMemoryNumpyLoader()

        logging.info(f"Loading {self.depth_path_file_list} with shared memory")
        self.depth_paths, depth_file_idx = self.depth_path_sm_loader.load(
            self.depth_path_file_list
        )

        assert (
            depth_file_idx == self.file_idx
        ), "Depth file and path file were not found at the same index"

    def _load_depth(self, image_path):
        """
        Returns:
            A (H, W, 1) tensor
        """
        with g_pathmgr.open(image_path, "rb") as fopen:
            # Depth is being saved as a .pt file instead
            # of as an image
            if image_path.endswith(".pt"):
                depth = torch.load(fopen).float()
            elif image_path.endswith(".png"):
                depth = Image.open(fopen)
                depth = torch.from_numpy(np.array(depth).astype(np.float32))
        return depth

    def _get_depth_path(self, idx):
        path = self._replace_path_prefix(
            self.depth_paths[idx],
            replace_prefix=self.remove_depth_prefix,
            new_prefix=self.new_depth_prefix,
        )
        path = self._replace_path_suffix(
            path,
            replace_suffix=self.remove_depth_suffix,
            new_suffix=self.new_depth_suffix,
        )
        return path

    @staticmethod
    def create_sample(idx, data, label, is_success):
        if isinstance(data, tuple):
            assert len(data) == 2
            image, depth = data[0], data[1]
            if isinstance(image, torch.Tensor):
                assert image.ndim == 3 and image.shape[0] == 3
            else:
                assert isinstance(image, Image.Image)

            assert (
                isinstance(depth, torch.Tensor)
                and depth.ndim == 3
                and depth.shape[0] == 1
            )
            return VisionDepthSample(
                vision=image,
                depth=depth,
                label=int(label),
                data_idx=idx,
                data_valid=is_success,
            )
        else:
            ImagePathDataset.create_sample(idx, data, label, is_success)

    def default_generator(self):
        image = get_mean_image(DEFAULT_SPATIAL_SIZE)
        depth = torch.zeros(
            (1, DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE), dtype=torch.float32
        )
        return torch.cat([tvf.to_tensor(image), depth], dim=0)

    def try_load_object(self, idx):
        image, is_success = super().try_load_object(idx)
        if self.load_image_as_tensor:
            image = tvf.to_tensor(image)
        if is_success:
            try:
                depth = self._load_depth(self._get_depth_path(idx))
                if depth.ndim == 2:
                    depth = depth[None, ...]  # (1, H, W)
            except Exception:
                logging.warning(
                    f"Couldn't load depth image: {self.depth_paths[idx]}. Exception:",
                    exc_info=True,
                )
                is_success = False

        if not is_success:
            image_with_depth = self.default_generator()
            image = image_with_depth[:3, ...]
            depth = image_with_depth[3:, ...]

        if self.concatenate_depth_and_rgb_channels:
            image_with_depth = torch.cat([image, depth], dim=0)  # (4, H, W)
        else:
            image_with_depth = (image, depth)

        return image_with_depth, is_success


class VideoPathDataset(PathDataset):
    def __init__(
        self,
        clip_sampler,
        frame_sampler,
        decoder,
        normalize_to_0_1,
        *args,
        decode_audio=False,
        # Number of bins in mel spectogram
        audio_num_mel_bins=128,
        audio_target_len=1024,
        decoder_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_sampler = clip_sampler
        self.frame_sampler = frame_sampler
        self.decoder = decoder
        self.normalize_to_0_1 = normalize_to_0_1
        self.decode_audio = decode_audio
        self.audio_num_mel_bins = audio_num_mel_bins
        self.audio_target_len = audio_target_len
        self.decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

    def _get_video_object(self, path):
        if self.decoder == "real_torchvision":
            local_path = g_pathmgr.get_local_path(path)
            assert not self.decode_audio, "Not supported"
            return RealTorchvisionVideoReader(local_path, **self.decoder_kwargs)
        return EncodedVideo.from_path(
            path,
            decoder=self.decoder,
            decode_audio=self.decode_audio,
            **self.decoder_kwargs,
        )

    def process_video_data(self, video_clip):
        frames = self.frame_sampler(video_clip)
        if self.normalize_to_0_1:
            frames = frames / 255.0  # since this is float, need 0-1
        return frames

    def process_audio_data(self, audio_clip):
        if not self.decode_audio:
            return audio_clip
        assert (
            "sample_rate" in self.decoder_kwargs
        ), "Must specify sample_rate when decoding audio. At least decord supports."
        return AudioPathDataset.waveform2melspec(
            waveform=audio_clip.unsqueeze(0),
            sample_rate=self.decoder_kwargs["sample_rate"],
            num_mel_bins=self.audio_num_mel_bins,
            target_length=self.audio_target_len,
        )

    def create_sample(self, idx, data, label, is_success):
        video_data, audio_data = data
        kwargs = {"label": label, "data_idx": idx, "data_valid": is_success}
        if self.decode_audio:
            return VisionAudioSample(vision=video_data, audio=audio_data, **kwargs)
        return VisionSample(vision=video_data, **kwargs)

    @staticmethod
    def get_clip_timepoints(clip_sampler, duration):
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(
                end, duration, annotation=None
            )
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints

    def load_object(self, path) -> List[torch.Tensor]:
        """
        Returns:
            A (C, T, H, W) tensor.
        """
        video = self._get_video_object(path)
        # Read out all clips in this video
        all_clips_timepoints = self.get_clip_timepoints(
            self.clip_sampler, video.duration
        )
        # TODO handle multiple clips.. for now, just pick one..
        # Instead of picking the middle, pick the first one. That's what
        # slowfast seems to do when I specify TEST.NUM_ENSEMBLE_VIEWS=1
        all_video = []
        all_audio = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                logging.error(
                    "Got a None clip. Make sure the clip timepoints "
                    "are long enough: %s",
                    clip_timepoints,
                )
                raise ValueError("No clip found")
            video_clip = self.process_video_data(clip["video"])
            audio_clip = self.process_audio_data(clip["audio"])
            all_video.append(video_clip)
            all_audio.append(audio_clip)
        # Right now most preprocessing layers can't really handle
        # multiple clips, so that's a TODO
        return all_video, all_audio

    def default_generator_video(self):
        # Not using image representations for 1 frame videos
        # if self.video_cfg.FRAME_SAMPLER.num_samples == 1:
        #     # Treat it as a image dataset, so return a image based dummy
        #     # image
        #     return super()._get_mean_image()
        # else:
        # FIXME: assumes a `_num_samples` and `_clips_per_video` attribute
        dummy = [
            (
                torch.ones(
                    (
                        3,
                        self.frame_sampler._num_samples,
                        DEFAULT_SPATIAL_SIZE,
                        DEFAULT_SPATIAL_SIZE,
                    )
                )
                * 0.5
            )
        ]
        if hasattr(self.clip_sampler, "_clips_per_video"):
            return dummy * self.clip_sampler._clips_per_video
        return dummy

    def default_generator_audio(self):
        return AudioPathDataset.dummy_audio_generator(
            self.audio_num_mel_bins, self.audio_target_len, self.clip_sampler
        )

    def default_generator(self):
        return self.default_generator_video(), self.default_generator_audio()


class AudioPathDataset(PathDataset):
    def __init__(
        self,
        clip_sampler,
        *args,
        num_mel_bins=128,
        target_length=1024,
        sample_rate=16000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_sampler = clip_sampler
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate

    @staticmethod
    def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
        # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
        waveform -= waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        # Convert to [mel_bins, num_frames] shape
        fbank = fbank.transpose(0, 1)
        # Pad to target_length
        n_frames = fbank.size(1)
        p = target_length - n_frames
        # if p is too large (say >20%), flash a warning
        if abs(p) / n_frames > 0.2:
            logging.warning(
                "Large gap between audio n_frames(%d) and "
                "target_length (%d). Is the audio_target_length "
                "setting correct?",
                n_frames,
                target_length,
            )
        # cut and pad
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
        elif p < 0:
            fbank = fbank[:, 0:target_length]
        # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
        # channel image
        fbank = fbank.unsqueeze(0)
        return fbank

    def load_object(self, path) -> List[torch.Tensor]:
        """
        Returns:
            A (C, T, H, W) tensor.
        """
        waveform, sr = torchaudio.load(path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        all_clips_timepoints = VideoPathDataset.get_clip_timepoints(
            self.clip_sampler, waveform.size(1) / self.sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(
                    clip_timepoints[1] * self.sample_rate
                ),
            ]
            waveform_melspec = self.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_clips.append(waveform_melspec)
        return all_clips

    def default_generator(self):
        return self.dummy_audio_generator(
            self.num_mel_bins, self.target_length, self.clip_sampler
        )

    @staticmethod
    def dummy_audio_generator(num_mel_bins, target_length, clip_sampler):
        dummy = [torch.zeros((1, num_mel_bins, target_length))]
        if hasattr(clip_sampler, "_clips_per_video"):
            return dummy * clip_sampler._clips_per_video
        return dummy

    @staticmethod
    def create_sample(idx, data, label, is_success):
        return AudioSample(audio=data, label=label, data_idx=idx, data_valid=is_success)


class PathDatasetWithTextLabels:
    def __init__(
        self,
        base_dataset: PathDataset,
        tokenizer: Any,
        label_names_file_list: List[str] = None,
        templates: List[str] = (),
    ):
        """To support text labels for datasets.
        Args:
            return_txt_labels (bool): If True, will add the text caption to the
                returned image. Will return a list [image, text]
            label_names_file_list (list of fpath). One of these must be a NPY file that contains a list
                of label names for each label ID as a list. So something like
                [
                    ["apple", "green apple", ...],  <-- All the ways you refer to label 0
                    ["car", "vehicle", ...],  <-- All the ways you refer to label 1
                    ...
                ]
                The number of names per label must be the same; since
                the features need to be batched later on.
            templates (list[str]): Each string should have a "{}" which will
                be replaced with the label name.
            mode (str): Set to val to return the all the label strings
                as well. These will be used to predict the class
                label.
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.label_names_file_list = label_names_file_list
        self.templates = templates
        self._load_label_names()

    def _load_label_names(self):
        assert self.label_names_file_list is not None and self.templates is not None
        self.label_strings = self.gen_label_strings(
            self.tokenizer, self.templates, self.label_names_file_list
        )
        if self.base_dataset.label_type == LABEL_TYPE_INT:
            assert np.issubdtype(self.base_dataset.labels.dtype, np.integer)
            assert max(self.base_dataset.labels) == len(self.label_strings) - 1
        elif self.base_dataset.label_type == LABEL_TYPE_CSV:
            # https://stackoverflow.com/questions/10790312/numpy-check-array-for-string-data-type#comment75655003_10790620
            assert self.base_dataset.labels.dtype.kind in {"U", "S"}
            assert (
                max(
                    flatten_list(
                        [
                            [int(e) for e in el.split(",")]
                            for el in self.base_dataset.labels
                        ]
                    )
                )
                == len(self.label_strings) - 1
            )
        else:
            raise NotImplementedError(
                f"label_type {self.base_dataset.label_type} not supported yet"
            )

    @staticmethod
    def gen_label_strings(tokenizer, templates, label_names_file_list):
        """
        Returns a list of len == #classes
        Each element is a list of length #label names for that class where elems are torch.Tensor of shape num_templates x context
            [
                [
                    torch.Tensor for label_name_1_class_1,
                    torch.Tensor for label_name_2_class_1,
                ],
                [
                    torch.Tensor for label_name_1_class_2,
                ],
                ...
            ]
        """
        label_names, _ = FileLoader.load(label_names_file_list)
        # label_names is a list/array of length num_classes
        # each element is a list of names for that class
        # e.g. [ ["dog", "puppy"], ["cat", "kitten, "tabby"]]
        if isinstance(templates, np.ndarray):
            assert templates.ndim == 1
            templates = templates.tolist()
        per_label_id_label_name_tokenized_templates = []
        for label_id in range(len(label_names)):
            formatted_templates = []
            for label_name in label_names[label_id]:
                tokenized_templates = tokenizer(
                    [template.format(label_name) for template in templates]
                )
                # formatted_templates = list of len #label names for this label_id
                formatted_templates.append(tokenized_templates)
            per_label_id_label_name_tokenized_templates.append(formatted_templates)
        return per_label_id_label_name_tokenized_templates

    def text_for_label(self, label: Union[int, List[int]]):
        if isinstance(label, List):
            # Randomly pick a target for this string
            # This is not ideal as all the labels are positive for this sample
            # But this is the easiest solution for now. Can't
            # append all the labels in the string as it is already tokenized
            label = label[np.random.randint(len(label))]
        label_strs = self.label_strings[label]
        # now randomly pick one of the label names associated with this label
        label_str = label_strs[np.random.randint(len(label_strs))]
        assert label_str.ndim == 2
        # now randomly pick one of the templates associated with this label name
        label_str = label_str[np.random.randint(len(label_str)), :]
        return label_str

    def __len__(self):
        return self.base_dataset.num_samples

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_class_with_text = UPGRADE_TO_TEXT_SAMPLE[type(sample)]
        return sample_class_with_text(
            **dataclass_as_dict(sample), text=self.text_for_label(sample.label)
        )