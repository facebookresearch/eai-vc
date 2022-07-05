import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import VisionDataset


class PathDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[str] = None,
        extra_transform: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        max_offset: int = 16,
        randomize_views: bool = False,
    ):
        assert (mean is None) == (std is None)
        super().__init__(root=root)

        self.folders = sorted(glob.glob(os.path.join(self.root, "*", "*")))
        self.files, self.idx_to_folder, file_idx = {}, {}, 0
        for folder_idx, folder in enumerate(self.folders):
            self.files[folder] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            for idx in range(len(self.files[folder])):
                self.idx_to_folder[file_idx + idx] = folder_idx
            file_idx += len(self.files[folder])

        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std
        self.max_offset = max_offset
        self.randomize_views = randomize_views

    def _get_image(self, folder, idx):
        path = self.files[folder][idx]
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
        return img, extra_img

    def __len__(self) -> int:
        return len(self.idx_to_folder)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        folder = self.folders[self.idx_to_folder[idx]]
        images = self.files[folder]
        max_offset = min(self.max_offset, len(images) - 1)
        min_offset = 1 if max_offset > 0 else 0
        offset = np.random.randint(min_offset, max_offset + 1)
        img1_idx = np.random.randint(0, len(images) - offset)
        img1, extra_img1 = self._get_image(folder, img1_idx)
        img2, extra_img2 = self._get_image(folder, img1_idx + offset)
        if self.randomize_views and np.random.rand() < 0.5:
            return img2, extra_img2, img1, extra_img1, offset
        return img1, extra_img1, img2, extra_img2, offset


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
        root="data/datasets/hm3d+gibson/v1/train",
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
        max_offset=16,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs1, extra_imgs1, imgs2, extra_imgs2, offsets in loader:
        print(imgs1.shape, extra_imgs1.shape, imgs2.shape, extra_imgs2.shape, offsets)
        break
    print()

    # print dataset info
    lengths = [len(item) for _, item in dataset.files.items()]
    print(dataset)
    print("num paths:  {:,}".format(len(lengths)))
    print("num images: {:,}".format(sum(lengths)))
    print("avg length: {:.1f}".format(sum(lengths) / len(lengths)))
    print("lengths:    [{}:{}]".format(min(lengths), max(lengths)))
    print()
