import glob
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class PathDataset(VisionDataset):
    def __init__(
        self, root: str, transform: Optional[str] = None, max_offset: int = 16
    ):
        super().__init__(root=root, transform=transform)
        self.folders = sorted(glob.glob(os.path.join(self.root, "*", "*")))
        self.files = {}
        for folder in self.folders:
            self.files[folder] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        self.transform = transform
        self.max_offset = max_offset

    def _get_image(self, folder, idx):
        path = self.files[folder][idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        folder = self.folders[idx]
        images = self.files[folder]
        offset = np.random.randint(0, self.max_offset)
        img_idx = np.random.randint(0, len(images) - offset)
        img1 = self._get_image(folder, img_idx)
        img2 = self._get_image(folder, img_idx + offset)
        return img1, img2, offset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = PathDataset(root="data/path-dataset/train")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs1, imgs2, offsets in loader:
        print(imgs1.shape, imgs2.shape, offsets.shape)
        break
