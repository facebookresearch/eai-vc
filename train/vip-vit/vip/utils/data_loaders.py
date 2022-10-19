# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds="ego4d"):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}{index:06}.jpg")
    else:
        try:
            return torchvision.io.read_image(f"{vid}/{index}.jpg")
        except:
            return torchvision.io.read_image(f"{vid}/{index}.png")


## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource="ego4d", datapath=None, num_workers=10, doaug="none"):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        assert datapath is not None
        self.doaug = doaug

        # Augmentations
        self.preprocess = torch.nn.Sequential(
            transforms.Resize(256), transforms.CenterCrop(224)
        )
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            )
        else:
            self.aug = lambda a: a

        # Load Data
        if "ego4d" == self.datasource:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)

    def _sample(self):
        # Sample a video from datasource
        if self.datasource == "ego4d":
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            try:
                vidlen = m["num_frames"]
                vid = m["directory"]
            except:
                vidlen = m["len"]
                vid = m["path"]
            # vid = m["path"].replace('surajn/data', 'aravraj/r3m_data')
        else:
            video_paths = glob.glob(f"{self.datapath}/[0-9]*")
            num_vid = len(video_paths)

            video_id = np.random.randint(0, int(num_vid))
            vid = f"{video_paths[video_id]}"

            # Video frames must be .png or .jpg
            vidlen = len(glob.glob(f"{vid}/*.png"))
            if vidlen == 0:
                vidlen = len(glob.glob(f"{vid}/*.jpg"))

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen - 2)
        end_ind = np.random.randint(start_ind + 1, vidlen)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip + 1, end_ind)

        # random frames for alignment evaluation
        begin_idx = 1  # hack right now
        s1_ind = np.random.randint(begin_idx + 1, vidlen)
        s0_ind = np.random.randint(begin_idx, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen + begin_idx)

        # Self-supervised reward (this is always -1)
        reward = float(s0_ind_vip == end_ind) - 1

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, self.datasource)
            img = get_ind(vid, end_ind, self.datasource)
            imts0_vip = get_ind(vid, s0_ind_vip, self.datasource)
            imts1_vip = get_ind(vid, s1_ind_vip, self.datasource)

            # allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
            # allims_aug = self.aug(allims / 255.0) * 255.0

            # im0 = allims_aug[0]
            # img = allims_aug[1]
            # imts0_vip = allims_aug[2]
            # imts1_vip = allims_aug[3]

            imts0 = get_ind(vid, s0_ind, self.datasource)
            imts1 = get_ind(vid, s1_ind, self.datasource)
            imts2 = get_ind(vid, s2_ind, self.datasource)
            allims = torch.stack(
                [im0, img, imts0_vip, imts1_vip, imts0, imts1, imts2], 0
            )
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0_vip = allims_aug[2]
            imts1_vip = allims_aug[3]

            imts0 = allims_aug[-3]
            imts1 = allims_aug[-2]
            imts2 = allims_aug[-1]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, self.datasource) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, self.datasource) / 255.0) * 255.0
            imts0_vip = (
                self.aug(get_ind(vid, s0_ind_vip, self.datasource) / 255.0) * 255.0
            )
            imts1_vip = (
                self.aug(get_ind(vid, s1_ind_vip, self.datasource) / 255.0) * 255.0
            )

            imts0 = self.aug(get_ind(vid, s0_ind, self.datasource) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, self.datasource) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, self.datasource) / 255.0) * 255.0

        # im = torch.stack([im0, img, imts0_vip, imts1_vip])
        im = torch.stack([im0, img, imts0_vip, imts1_vip, imts0, imts1, imts2])
        im = self.preprocess(im)
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()
