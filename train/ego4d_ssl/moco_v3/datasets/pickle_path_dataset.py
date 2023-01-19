from __future__ import annotations
import os, pickle, gc
import torchvision.datasets as datasets
import torch.utils.data as data
import torch, torchvision
import numpy as np
from typing import Union, Tuple, List, Dict
from PIL import Image


class PicklePathsDataset(data.Dataset):
    """Dataset that reads frames generated from trajectories"""

    def __init__(
        self,
        root_dir: str,
        frameskip: int = 1,
        transforms: Union[torchvision.transforms.Compose, List, None] = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Creates a dataset using frames from sub-directories in root directory.
        Frames are expected to be organized as:
        root
        | -- folder_1 (name of task/scene)
            | -- subfolder_1 (traj_i)
                | -- image_1 (frame_j.jpg)
                | -- image_2
        """
        data.Dataset.__init__(self)
        assert os.path.isdir(root_dir)
        self.root_dir = root_dir
        # get a list of all pickle files in root directory
        tasks = next(os.walk(root_dir))[1]

        # store the transforms
        self.transforms = transforms
        if self.transforms is None:
            print("\n PicklePathsDataset requires an input transforms. Aborting run.")
            quit()
        elif type(self.transforms) == list:
            assert len(self.transforms) == 2
            for t in self.transforms:
                assert isinstance(t, torchvision.transforms.Compose)
        elif isinstance(self.transforms, torchvision.transforms.Compose):
            self.transforms = [self.transforms, self.transforms]
        else:
            print("Unsupported input transformations in PicklePathsDataset.")
            quit()

        # maintain a frame buffer in memory and populate from dataset
        self.frame_buffer = []
        for task in tasks:
            print("Currently loading task: %s" % task) if debug_mode else {}
            task_root = os.path.join(root_dir, task)
            trajectories = next(os.walk(task_root))[1]
            for traj in trajectories:
                traj_root = os.path.join(task_root, traj)
                frames = os.listdir(traj_root)
                for timestep, frame in enumerate(frames):
                    if timestep % frameskip == 0:
                        frame_meta_data = {
                            "path": os.path.join(traj_root, frame),
                            "task": task,
                            "traj": traj,
                            "time": timestep,
                        }
                        self.frame_buffer.append(frame_meta_data)

        # print messages
        print("\n Successfully loaded dataset from root_dir: %s" % root_dir)
        print("\n Dataset size is: %i" % len(self.frame_buffer))

    def __getitem__(self, index: int) -> Dict:
        frame = self.frame_buffer[index]
        frame = Image.open(frame["path"])
        # compute two different views/augmentations of the same image
        im1 = self.transforms[0](frame)
        im2 = self.transforms[1](frame)
        out = {
            "input1": im1,
            "input2": im2,
            "meta": dict(),
        }
        return out

    def __len__(self) -> int:
        return len(self.frame_buffer)


class TimeContrastiveDataset(PicklePathsDataset):
    """
    Dataset that reads frames generated from trajectories
    and provides samples for time contrastive learning.
    """

    def __getitem__(self, index: int) -> Dict:
        frame = self.frame_buffer[index]
        next_frame = self.frame_buffer[index + 1]
        if not (
            frame["task"] == next_frame["task"] and frame["traj"] == next_frame["traj"]
        ):
            # take previous frame since we cross over into different traj or scene
            next_frame = self.frame_buffer[index - 1]
        im1 = self.transforms[0](Image.open(frame["path"]))
        im2 = self.transforms[1](Image.open(next_frame["path"]))
        out = {
            "input1": im1,
            "input2": im2,
            "meta": dict(task=frame["task"], traj=frame["traj"]),
        }
        return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import RandomResizedCrop, ToTensor
    from tqdm import tqdm
    import time

    batch_size, num_workers = 64, 32

    dataset = PicklePathsDataset(
        root_dir="/checkpoint/maksymets/eaif/datasets/metaworld-expert-v0.1/",
        # image_key="images",
        frameskip=5,
        transforms=torchvision.transforms.Compose(
            [
                RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
                ToTensor(),
            ]
        ),
        debug_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    t0 = time.time()
    for idx, batch in tqdm(enumerate(data_loader)):
        q, k = batch["input1"], batch["input2"]
    print(
        "\n PicklePathsDataset \n"
        "Generated %i mini-batches of size %i | took time = %2.3f seconds \n"
        % (idx + 1, batch_size, time.time() - t0)
    )

    dataset = PicklePathsDataset(
        root_dir="/checkpoint/maksymets/eaif/datasets/hm3d+gibson/v1/train/",
        frameskip=5,
        transforms=torchvision.transforms.Compose(
            [
                RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
                ToTensor(),
            ]
        ),
        debug_mode=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    t0 = time.time()
    for idx, batch in tqdm(enumerate(data_loader)):
        q, k = batch["input1"], batch["input2"]
        if idx >= 10000:
            break
    print(
        "\n PicklePathsDataset \n"
        "Generated %i mini-batches of size %i | took time = %2.3f seconds \n"
        % (idx + 1, batch_size, time.time() - t0)
    )
