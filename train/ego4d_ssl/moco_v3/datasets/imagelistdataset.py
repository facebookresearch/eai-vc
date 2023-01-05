import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transF
import torch.utils.data as data
import numpy as np
import pdb
import itertools
import torch
import os
from scipy.stats import gamma
from collections import defaultdict


class ImageListDataset(data.Dataset):
    """Dataset that reads videos"""

    def __init__(self, list_fname, base_transform1, base_transform2, max_len=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert os.path.exists(list_fname), "{} does not exist".format(list_fname)
        assert type(list_fname) == str
        if list_fname.endswith(".txt"):
            with open(list_fname, "r") as f:
                filedata = f.read().splitlines()
        elif list_fname.endswith(".npy"):
            filedata = list(np.load(list_fname))
        else:
            print("Unknown file format in ImageListDataset")
            exit()

        if max_len is not None:
            filedata = filedata[:max_len]
        self.pair_filelist = [(d.split(" ")[0], d.split(" ")[0]) for d in filedata]

        if isinstance(base_transform1, list):
            base_transform1 = transforms.Compose(base_transform1)
        if isinstance(base_transform2, list):
            base_transform2 = transforms.Compose(base_transform2)
        self.base_transform1, self.base_transform2 = base_transform1, base_transform2

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, fname2 = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        im2 = datasets.folder.pil_loader(fname2)
        im1 = self.base_transform1(im1)
        im2 = self.base_transform2(im2)
        out = {
            "input1": im1,
            "input2": im2,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        RandomApply,
        RandomHorizontalFlip,
        RandomResizedCrop,
        ToTensor,
    )
    import time

    batch_size, num_workers = 16, 16

    ego4d_dataset = ImageListDataset(
        list_fname="/checkpoint/yixinlin/eaif/datasets/ego4d.npy",
        base_transform1=[RandomHorizontalFlip(), ToTensor()],
        base_transform2=[RandomHorizontalFlip(), ToTensor()],
        max_len=1600,
    )
    ego4d_loader = torch.utils.data.DataLoader(
        ego4d_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    t0 = time.time()
    for idx, batch in enumerate(ego4d_loader):
        q, k = batch["input1"], batch["input2"]
    print(
        "\n Ego4D dataset \n"
        "Generated %i mini-batches of size %i | took time = %2.3f seconds \n"
        % (idx + 1, batch_size, time.time() - t0)
    )

    hm3d_dataset = ImageListDataset(
        list_fname="/checkpoint/yixinlin/eaif/datasets/hm3d+gibson.npy",
        base_transform1=[RandomHorizontalFlip(), ToTensor()],
        base_transform2=[RandomHorizontalFlip(), ToTensor()],
        max_len=1600,
    )
    hm3d_loader = torch.utils.data.DataLoader(
        hm3d_dataset, num_workers=num_workers, batch_size=batch_size
    )

    t0 = time.time()
    for idx, batch in enumerate(hm3d_loader):
        q, k = batch["input1"], batch["input2"]
    print(
        "\n HM3D dataset \n"
        "Generated %i mini-batches of size %i | took time = %2.3f seconds \n"
        % (idx + 1, batch_size, time.time() - t0)
    )
