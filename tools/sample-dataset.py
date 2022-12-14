""" Script to subsample a dataset.

Samples folders assuming the following directory structure:

root
| -- folder_1
     | -- subfolder_1
        | -- image_1
        | -- image_2
        ...
        | -- image_N
     | -- subfolder_2
     ...
     | -- subfolder_M
...
| -- folder_P

This script assumes each folder contains a similar number of images.
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import tqdm


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="path to original dataset")
    parser.add_argument("dst", type=str, help="path to new dataset")
    parser.add_argument(
        "-p", default=0.05, type=float, help="sampling percentage (default: 0.05)"
    )
    args = parser.parse_args()

    # check command line arguments
    args.src = Path(args.src)
    assert args.src.exists()

    args.dst = Path(args.dst)
    args.dst.mkdir(parents=True, exist_ok=False)

    # set random seeds
    random.seed(0)
    np.random.seed(0)

    print("=" * 80)

    # get folders
    scene_folders = sorted([f for f in args.src.glob("*") if f.is_dir()])
    print("found {:,} folders".format(len(scene_folders)))

    # get images in each folder
    all_files = {}
    for folder in scene_folders:
        all_files[folder.name] = sorted(folder.glob("*/*.jpg"))

    # count the total number of images
    all_counts = [len(all_files[k]) for k in all_files]
    print("found {:,} files".format(sum(all_counts)))

    # print stats
    print(
        "  min: {} mean: {:0.1f} ({:0.1f}) max: {}".format(
            np.min(all_counts),
            np.mean(all_counts),
            np.std(all_counts),
            np.max(all_counts),
        )
    )

    print("=" * 80)

    # sample folders
    num_sampled_folders = int(np.ceil(args.p * len(scene_folders)))
    np.random.shuffle(scene_folders)
    sampled_folders = scene_folders[:num_sampled_folders]
    print("sampled {} of {} folders".format(len(sampled_folders), len(scene_folders)))

    # get images in each sampled folder
    sampled_files = {}
    for folder in sampled_folders:
        sampled_files[folder.name] = sorted(folder.glob("*/*.jpg"))

    # count the total number of sampled images
    sampled_counts = [len(sampled_files[k]) for k in sampled_files]
    print("sampled {:,} of {:,} files".format(sum(sampled_counts), sum(all_counts)))

    # print stats
    print(
        "  min: {} mean: {:0.1f} ({:0.1f}) max: {}".format(
            np.min(sampled_counts),
            np.mean(sampled_counts),
            np.std(sampled_counts),
            np.max(sampled_counts),
        )
    )
    print("=" * 80)

    print("desired percentage: {:0.2f}%".format(100 * args.p))
    print(
        "actual percentage: {:0.2f}%".format(
            100 * sum(sampled_counts) / sum(all_counts)
        )
    )

    print("=" * 80)

    # copy folders
    for src in tqdm.tqdm(sampled_folders):
        dst = args.dst / src.relative_to(args.src)
        shutil.copytree(src, dst)

    print("=" * 80)

    # get new folders
    new_folders = sorted([f for f in args.dst.glob("*") if f.is_dir()])
    print("found {:,} folders".format(len(new_folders)))

    # get images in each new folder
    new_files = {}
    for folder in new_folders:
        new_files[folder.name] = sorted(folder.glob("*/*.jpg"))

    # count the total number of new images
    new_counts = [len(new_files[k]) for k in new_files]
    print("found {:,} files".format(sum(new_counts)))

    # print stats (should match above)
    print(
        "  min: {} mean: {:0.1f} ({:0.1f}) max: {}".format(
            np.min(new_counts),
            np.mean(new_counts),
            np.std(new_counts),
            np.max(new_counts),
        )
    )

    print("=" * 80)

    # save list of file paths
    path_file_list = []
    for key in new_files:
        path_file_list.extend(new_files[key])
    path_file_list = sorted(path_file_list)
    np.save(args.dst / "path_file_list.npy", path_file_list)


if __name__ == "__main__":
    main()
