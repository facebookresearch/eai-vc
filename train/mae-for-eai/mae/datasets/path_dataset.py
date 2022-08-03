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
        root: List[str],
        transform: Optional[str] = None,
        extra_transform: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super().__init__(root=root)

        self.files = []
        for folder in self.root:
            self.files.extend(sorted(glob.glob(os.path.join(folder, "*", "*", "*.jpg"))))
            self.files.extend(sorted(glob.glob(os.path.join(folder, "*", "*", "*.png"))))

        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std
        assert (mean is None) == (std is None)

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
