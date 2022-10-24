import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
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
        max_offset: int = 16,
        randomize_views: bool = False,
    ):
        assert (mean is None) == (std is None)
        super().__init__(root=root)

        self.folders = []
        for folder in self.root:
            self.folders.extend(sorted(glob.glob(os.path.join(folder, "*", "*"))))

        self.files, self.idx_to_folder, file_idx = {}, {}, 0
        for folder_idx, folder in enumerate(self.folders):
            self.files[folder] = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            self.files[folder].extend(sorted(glob.glob(os.path.join(folder, "*.png"))))
            for idx in range(len(self.files[folder])):
                self.idx_to_folder[file_idx + idx] = folder_idx
            file_idx += len(self.files[folder])

        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std
        self.max_offset = max_offset  # not used
        self.randomize_views = randomize_views

    def _get_image_raw(self, folder, idx):
        path = self.files[folder][idx]
        img = Image.open(path).convert("RGB")
        extra_img = img.copy()
        img, extra_img = TF.to_tensor(img), TF.to_tensor(extra_img)
        return img, extra_img

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

        vidlen = len(images)
        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen - 2)
        end_ind = np.random.randint(start_ind + 1, vidlen)
        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip + 1, end_ind)
        offset = s0_ind_vip - start_ind

        # random frames for alignment evaluation
        begin_idx = 0  # hack right now
        s1_ind = np.random.randint(begin_idx + 1, vidlen)
        s0_ind = np.random.randint(begin_idx, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen + begin_idx)

        # TODO: How can we consistentize all transform with a video sequence?
        # o_initial, extra_o_initial = self._get_image(folder, start_ind)
        # o_final, extra_o_final = self._get_image(folder, end_ind)
        # o_middle, extra_o_middle =  self._get_image(folder, s0_ind_vip)
        # o_middle2, extra_o_middle2 = self._get_image(folder, s1_ind_vip)
        # obs = torch.stack([o_initial, o_final, o_middle, o_middle2], 0)
        # obs_extra = torch.stack([extra_o_initial, extra_o_final, extra_o_middle, extra_o_middle2])

        obs = []
        obs_extra = []
        for idx in [start_ind, end_ind, s0_ind_vip, s1_ind_vip, s0_ind, s1_ind, s2_ind]:
        # for idx in [start_ind, end_ind, s0_ind_vip, s1_ind_vip]:
            o, o_extra = self._get_image_raw(folder, idx)
            obs.append(o)
            obs_extra.append(o_extra)
        obs = torch.stack(obs, 0)
        obs_extra = torch.stack(obs_extra, 0)
        obs_transformed = self.transform(obs)
        obs_extra_transformed = self.transform(obs_extra)

        return obs_transformed, obs_extra_transformed, offset


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
        root=[
            # "data/datasets/hm3d+gibson/v1/train",
            # "data/datasets/real-estate-10k-frames-v0",
            "/checkpoint/yixinlin/eaif/datasets/ego4d"
        ],
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
    for imgs1, extra_imgs1, offsets in loader:
        print(imgs1.shape, extra_imgs1.shape, offsets)
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
