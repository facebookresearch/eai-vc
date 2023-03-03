#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import imageio
import numpy as np
import tqdm
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser("verify sp dataset")
    parser.add_argument("root", type=str, help="dataset root directory")
    parser.add_argument("--verify", action="store_true", help="verify dataset")
    parser.add_argument("--view", action="store_true", help="view a random sequence")
    parser.add_argument("--fname", default="temp.mp4", type=str, help="output filename")
    return parser.parse_args()


def verify(args):
    count = 0
    folders = sorted(glob.glob(os.path.join(args.root, "*", "*")))
    for folder in tqdm.tqdm(folders):
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for path in files:
            Image.open(path)
        count += len(files)
    print("verified {:,} files".format(count))


def view(args):
    folders = sorted(glob.glob(os.path.join(args.root, "*", "*")))
    folder = np.random.choice(folders)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    images = [np.array(Image.open(path)) for path in files]
    writer = imageio.get_writer(args.fname, fps=5, quality=5)
    for img in images:
        writer.append_data(img)
    writer.close()
    print(f"saved {folder} to: {os.path.abspath(args.fname)}")


if __name__ == "__main__":
    args = get_args()
    if args.verify:
        verify(args)
    if args.view:
        view(args)
