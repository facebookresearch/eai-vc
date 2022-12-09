import glob
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import VisionDataset


class PathDataset(VisionDataset):
    def __init__(
        self,
        root: List[str],
        transform: Optional[str] = None,
        extra_transform: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super().__init__(root=root)

        self.files = []
        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std
        assert (mean is None) == (std is None)

        self.get_files()

    def get_files(self):
        for folder in self.root:
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "*", "*", "*.jpg")))
            )
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "*", "*", "*.png")))
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.extra_transform is not None:
            extra_img = self.extra_transform(img)
        else:
            extra_img = img.copy()
        img, extra_img = TF.to_tensor(img), TF.to_tensor(extra_img)
        if self.mean is not None and self.std is not None:
            img = TF.normalize(img, self.mean, self.std)
            extra_img = TF.normalize(extra_img, self.mean, self.std)
        return img, extra_img, 0


class PathDatasetWithManifest(PathDataset):
    def __init__(self, sample_size=None, every_k=None, *args, **kwargs):
        """
        `sample_size` is either None or a list of integers corresponding to
        each manifest file. The datasets are sampled to the size of each
        corresponding integer; if the integer is -1, then the full dataset is
        retained. `sample_size is None` is equivalent to -1 for all datasets.

        `every_k` is either None or a list of integers corresponding to a
        simply picking every k-th file in each dataset manifest.
        """
        self.sample_size = sample_size
        self.every_k = every_k
        super().__init__(*args, **kwargs)
        assert sample_size is None or len(sample_size) == len(self.root)
        assert every_k is None or len(every_k) == len(self.root)
        assert not (sample_size is not None and every_k is not None)

    def get_files(self):
        manifest_files = self.root

        def read_manifest(manifest_file):
            with open(manifest_file) as f:
                return [line.rstrip() for line in f]

        self.total_file_dict = {
            manifest_file: read_manifest(manifest_file) for manifest_file in self.root
        }

        for i, manifest_file in enumerate(manifest_files):
            files = self.total_file_dict[manifest_file]
            print(f"Manifest file {manifest_file} has {len(files):,} files")
            if self.sample_size and self.sample_size[i] != -1:
                files = random.sample(files, self.sample_size[i])
                print(
                    f"Sampling dataset of {manifest_file} so that corresponding dataset has {len(files):,} files"
                )
            elif self.every_k and self.every_k[i] > 1:
                files = files[:: self.every_k[i]]
                print(
                    f"Using every {self.every_k[i]} file in {manifest_file} so that corresponding dataset has {len(files):,} files"
                )
            self.files.extend(files)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        RandomApply,
        RandomHorizontalFlip,
        RandomResizedCrop,
    )

    dataset = PathDataset(
        root="/checkpoint/karmeshyadav/hm3d+gibson/v1/train",
        transform=Compose(
            [
                RandomResizedCrop(224, (0.2, 1.0), interpolation=3),
                RandomHorizontalFlip(),
            ]
        ),
        extra_transform=RandomApply(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
        ),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for imgs1, extra_imgs1, _ in loader:
        print(imgs1.shape, extra_imgs1.shape)
        break
    print()
