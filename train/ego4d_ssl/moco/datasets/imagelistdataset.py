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


def longtail_class_distrib(list_fname=None, seed=0, num_classes=-1):
    """TODO: Docstring for longtail_class_distrib.

    :list_fname: TODO
    :seed: TODO
    :num_classes: TODO
    :returns: TODO

    """
    # Generate Gamma distribution
    rv = gamma(3, loc=-4, scale=2.0)
    if list_fname is not None:
        lab_to_fnames = defaultdict(list)
        with open(list_fname, "r") as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(" ")[1])].append(line.split(" ")[0])
        labels = list(lab_to_fnames.keys())
    else:
        labels = list(range(num_classes))
    # Seed controls which classes are in tail
    np.random.seed(seed)
    np.random.shuffle(labels)
    distrib = np.array(
        [rv.pdf(li * 18 / 1000.0) / rv.pdf(0) for li in range(len(labels))]
    )
    class_distrib = distrib[np.argsort(labels)]
    return class_distrib


class ImageListDataset(data.Dataset):
    """Dataset that reads videos"""

    def __init__(self, list_fname, transforms=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert os.path.exists(list_fname), "{} does not exist".format(list_fname)
        with open(list_fname, "r") as f:
            filedata = f.read().splitlines()
            self.pair_filelist = [(d.split(" ")[0], d.split(" ")[0]) for d in filedata]
            print(self.pair_filelist[:10])

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.std = torch.Tensor(self.transforms[0].transforms[-1].std).view(3, 1, 1)
        self.mean = torch.Tensor(self.transforms[0].transforms[-1].mean).view(3, 1, 1)

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, fname2 = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        im2 = datasets.folder.pil_loader(fname2)
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im1)
            im2 = transform(im2)
        meta["transind1"] = i
        meta["transind2"] = i

        out = {
            "input1": im1,
            "input2": im2,
            "meta": meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)


class LongTailImageListDataset(ImageListDataset):
    """Docstring for LongTailImageListDataset."""

    def __init__(self, list_fname, transforms=None, seed=1992):
        """TODO: to be defined."""
        ImageListDataset.__init__(self, list_fname, transforms=transforms)

        # Generate Gamma distribution
        rv = gamma(3, loc=-4, scale=2.0)

        lab_to_fnames = defaultdict(list)
        with open(list_fname, "r") as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(" ")[1])].append(line.split(" ")[0])

        # Seed controls which classes are in tail
        np.random.seed(seed)
        labels = list(lab_to_fnames.keys())
        np.random.shuffle(labels)

        # Sample images for each class
        max_im_per_lab = max([len(v) for v in lab_to_fnames.values()])
        for li, lab in enumerate(labels):
            # Magic numbers
            num = int(rv.pdf(li * 18 / 1000.0) * max_im_per_lab / rv.pdf(0))
            replace = len(lab_to_fnames[lab]) < num
            lab_to_fnames[lab] = np.random.choice(
                lab_to_fnames[lab], size=num, replace=replace
            )

        self.pair_filelist = [
            (v, v) for lab_fnames in lab_to_fnames.values() for v in lab_fnames
        ]
        print(self.pair_filelist[:10])


class UniformImageListDataset(ImageListDataset):
    """Docstring for UniformImageListDataset."""

    def __init__(self, list_fname, transforms=None, seed=1992, num_images=1000):
        """TODO: to be defined."""
        ImageListDataset.__init__(self, list_fname, transforms=transforms)

        lab_to_fnames = defaultdict(list)
        with open(list_fname, "r") as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(" ")[1])].append(line.split(" ")[0])

        # Seed controls which images are sampled
        np.random.seed(seed)
        labels = list(lab_to_fnames.keys())
        num_per_lab = num_images // len(labels)

        # Sample images for each class
        for li, lab in enumerate(labels):
            replace = len(lab_to_fnames[lab]) < num_per_lab
            lab_to_fnames[lab] = np.random.choice(
                lab_to_fnames[lab], size=num_per_lab, replace=replace
            )

        self.pair_filelist = [
            (v, v) for lab_fnames in lab_to_fnames.values() for v in lab_fnames
        ]
        print(self.pair_filelist[:10])


class ImageListStandardDataset(data.Dataset):
    """Dataset that reads videos"""

    def __init__(self, list_fname, transform=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert os.path.exists(list_fname), "{} does not exist".format(list_fname)
        with open(list_fname, "r") as f:
            filedata = f.read().splitlines()
            self.pair_filelist = [
                (d.split(" ")[0], int(d.split(" ")[1])) for d in filedata
            ]

        self.transform = transform

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, target = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        if self.transform is not None:
            im1 = self.transform(im1)

        out = {
            "input": im1,
            "target": torch.tensor(target),
            "fname": fname1,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)
