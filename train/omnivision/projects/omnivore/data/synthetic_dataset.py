import logging
from typing import List, Optional

import numpy as np

import torch
from omnivore.data.api import VisionSample, VisionTextHashtagSample, VisionTextSample
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, tensor_shape, length, label=0, value=1, transforms=None) -> None:
        self.tensor = torch.full(tuple(tensor_shape), float(value))
        self.label = label
        self.length = length
        self.transforms = transforms

    def __len__(self) -> int:
        return self.length

    def apply_transforms(self, sample):
        if self.transforms is None:
            return sample
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx) -> VisionSample:
        return self.apply_transforms(
            VisionSample(
                vision=self.tensor,
                label=self.label,
                data_idx=idx,
                data_valid=True,
            )
        )


def generate_static_image(height: int, width: int):
    return Image.new("RGB", (height, width))


def generate_random_image(seed: int, height: int, width: int):
    rng = np.random.RandomState(seed)
    noise_size = max(height, width) // 16
    gaussian_kernel_radius = rng.randint(noise_size // 2, noise_size * 2)
    img = Image.fromarray((255 * rng.rand(noise_size, noise_size, 3)).astype(np.uint8))
    img = img.resize((height, width))
    img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_kernel_radius))
    return img


def generate_image(seed: int, height: int, width: int, random_image: bool):
    if random_image:
        return generate_random_image(
            seed=seed,
            height=height,
            width=width,
        )
    else:
        return generate_static_image(
            height=height,
            width=width,
        )


class SyntheticVisionSampleDataset(Dataset):
    """
    Creates VisionSample where the image can be generated randomly
    """

    def __init__(
        self,
        visual_tensor_shape: List[int],
        length: int,
        random_image: bool = True,
        num_classes: int = 1,
        transforms=None,
    ) -> None:
        self.visual_tensor_shape = visual_tensor_shape
        self.length = length
        self.random_image = random_image
        self.transforms = transforms
        self.num_classes = num_classes
        logging.info(f"Created synthetic dataset of length: {self.length}")

    def __len__(self) -> int:
        return self.length

    def apply_transforms(self, sample):
        if self.transforms is None:
            return sample
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx) -> VisionTextSample:
        visual_data = generate_image(
            seed=idx,
            height=self.visual_tensor_shape[-2],
            width=self.visual_tensor_shape[-1],
            random_image=self.random_image,
        )
        label = idx % self.num_classes
        return self.apply_transforms(
            VisionSample(
                vision=visual_data,
                label=label,
                data_idx=idx,
                data_valid=True,
            )
        )


class SyntheticVisionTextDataset(Dataset):
    """
    Creates VisionTextSample where the image can be generated randomly
    """

    def __init__(
        self,
        visual_tensor_shape: List[int],
        length: int,
        random_image: bool = True,
        num_classes: int = 1,
        caption: str = "a list of words",
        transforms=None,
    ) -> None:
        self.visual_tensor_shape = visual_tensor_shape
        self.num_classes = num_classes
        self.length = length
        self.random_image = random_image
        self.caption = caption
        self.transforms = transforms

    def __len__(self) -> int:
        return self.length

    def apply_transforms(self, sample):
        if self.transforms is None:
            return sample
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx) -> VisionTextSample:
        visual_data = generate_image(
            seed=idx,
            height=self.visual_tensor_shape[-2],
            width=self.visual_tensor_shape[-1],
            random_image=self.random_image,
        )
        label = idx % self.num_classes
        return self.apply_transforms(
            VisionTextSample(
                vision=visual_data,
                data_idx=idx,
                label=label,
                data_valid=True,
                text=self.caption,
            )
        )


class SyntheticVisionTextHashtagDataset(Dataset):
    """
    Creates VisionTextSample where the image can be generated randomly
    """

    def __init__(
        self,
        visual_tensor_shape: List[int],
        length: int,
        random_image: bool = True,
        num_classes: int = 1,
        captions: Optional[List[str]] = None,
        hashtags: str = "cat,dog",
        transforms=None,
    ) -> None:
        self.visual_tensor_shape = visual_tensor_shape
        self.num_classes = num_classes
        self.length = length
        self.random_image = random_image
        self.captions = captions or ["a list of words"]
        self.hashtags = hashtags
        self.transforms = transforms

    def __len__(self) -> int:
        return self.length

    def apply_transforms(self, sample):
        if self.transforms is None:
            return sample
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __getitem__(self, idx) -> VisionTextSample:
        visual_data = generate_image(
            seed=idx,
            height=self.visual_tensor_shape[-2],
            width=self.visual_tensor_shape[-1],
            random_image=self.random_image,
        )
        label = idx % self.num_classes
        caption = self.captions[idx % len(self.captions)]
        return self.apply_transforms(
            VisionTextHashtagSample(
                vision=visual_data,
                data_idx=idx,
                label=label,
                data_valid=True,
                text=caption,
                hashtags=self.hashtags,
            )
        )
