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
        self.files, self.idx_to_folder, file_idx = {}, {}, 0
        for folder_idx, folder in enumerate(self.folders):
            self.files[folder] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            for idx in range(len(self.files[folder])):
                self.idx_to_folder[file_idx + idx] = folder_idx
            file_idx += len(self.files[folder])
        self.transform = transform
        self.max_offset = max_offset

    def _get_image(self, folder, idx):
        path = self.files[folder][idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.idx_to_folder)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        folder = self.folders[self.idx_to_folder[idx]]
        images = self.files[folder]
        max_offset = min(self.max_offset, len(images) - 1)
        min_offset = 1 if max_offset > 0  else 0
        offset = np.random.randint(min_offset, max_offset + 1)
        img1_idx = np.random.randint(0, len(images) - offset)
        img1 = self._get_image(folder, img1_idx)
        img2 = self._get_image(folder, img1_idx + offset)
        return img1, img2, offset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor

    dataset = PathDataset(
        root="data/datasets/hm3d+gibson/v1/train", transform=ToTensor()
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs1, imgs2, offsets in loader:
        print(imgs1.shape, imgs2.shape, offsets)
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
