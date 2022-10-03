import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torchvision
from omnivore.data.transforms.image_rgbd import DepthNorm, DepthToInverseDepth
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,
    help="Root dir which contains Training and Validation folders",
    default=None,
)
parser.add_argument(
    "--split", type=str, choices=["train", "val", "train+val"], required=True
)
parser.add_argument("--dst_dir", type=str, required=True)
parser.add_argument(
    "--num_processes", type=int, default=10, help="num processes for mp.Pool"
)
parser.add_argument(
    "--only_stats",
    action="store_true",
    default=False,
)


SPLIT_TO_DIR = {
    "train": "Training",
    "val": "Validation",
}


def get_files(dirname, ext):
    filelist = glob.glob(dirname + f"/*{ext}")
    return filelist


def unroll_list(list_of_lists):
    flat_list = []
    for el in list_of_lists:
        flat_list.extend(el)
    return flat_list


def read_file_and_compute_stats(fname, transform):
    depth = Image.open(fname)
    depth = np.array(depth).astype(np.float32)
    depth = torch.from_numpy(depth)
    depth = depth[..., None].permute(2, 0, 1)
    disparity = transform(depth)
    curr_mean = disparity.mean().item()
    curr_std = disparity.std().item()
    curr_max = disparity.max().item()
    curr_min = disparity.min().item()

    return [curr_mean, curr_std, curr_max, curr_min]


def compute_mean_disparity(depth_filenames, transform, num_processes=10):
    # take a subset
    subset_inds = np.linspace(
        0, len(depth_filenames), num=2000, endpoint=False, dtype=np.int64
    )
    depth_filenames = [depth_filenames[x] for x in subset_inds]

    with Pool(num_processes) as p:
        stats = p.map(
            partial(read_file_and_compute_stats, transform=transform), depth_filenames
        )

    means = [x[0] for x in stats]
    mean_disparity = sum(means)
    mean_disparity /= len(stats)

    # we compute std as avg of stds. This isn't mathematically correct, but is done in other datasets.
    stds = [x[1] for x in stats]
    std_disparity = sum(stds)
    std_disparity /= len(stats)

    maxs = [x[2] for x in stats]
    mins = [x[3] for x in stats]
    maxs = max(maxs)
    mins = min(mins)

    print(f"Using transform {transform}. Max {maxs}. Min {mins}")

    return mean_disparity, std_disparity, len(depth_filenames)


def main():
    args = parser.parse_args()

    if args.split == "train+val":
        splits = args.split.split("+")
    else:
        splits = [args.split]

    image_filenames = []
    highres_depth_filenames = []
    lowres_depth_filenames = []

    for split in splits:
        split_dir = os.path.join(args.root_dir, SPLIT_TO_DIR[split])
        scene_dirs = os.listdir(split_dir)

        rgb_dirs = [os.path.join(split_dir, x, "wide") for x in scene_dirs]
        with Pool(args.num_processes) as p:
            filelist = p.map(partial(get_files, ext=".png"), rgb_dirs)

        rgb_filelist = unroll_list(filelist)
        print(f"Found {len(rgb_filelist)} files in split {split}")
        highres_depth_filelist = [
            x.replace("wide", "highres_depth") for x in rgb_filelist
        ]
        lowres_depth_filelist = [
            x.replace("wide", "lowres_depth") for x in rgb_filelist
        ]
        image_filenames.extend(rgb_filelist)
        highres_depth_filenames.extend(highres_depth_filelist)
        lowres_depth_filenames.extend(lowres_depth_filelist)
    print(f"Total of {len(image_filenames)} files in splits {splits}")

    image_filenames = np.array(image_filenames)
    highres_depth_filenames = np.array(highres_depth_filenames)
    lowres_depth_filenames = np.array(lowres_depth_filenames)
    fake_labels = np.arange(len(image_filenames)).astype(np.int64)
    if not args.only_stats:
        np.save(os.path.join(args.dst_dir, f"{args.split}_images.npy"), image_filenames)
        np.save(
            os.path.join(args.dst_dir, f"{args.split}_highres_depth_pngs.npy"),
            highres_depth_filenames,
        )
        np.save(
            os.path.join(args.dst_dir, f"{args.split}_lowres_depth_pngs.npy"),
            lowres_depth_filenames,
        )
        np.save(
            os.path.join(args.dst_dir, f"{args.split}_fake_labels.npy"), fake_labels
        )

        fake_label_strings = [f"{x:09d}" for x in fake_labels]
        fake_label_strings = np.array(fake_label_strings)[..., None]
        np.save(
            os.path.join(args.dst_dir, f"{args.split}_fake_label_strings.npy"),
            fake_label_strings,
        )

        # Add dataset identifier
        fake_label_strings = [f"ARKit {x:09d}" for x in fake_labels]
        fake_label_strings = np.array(fake_label_strings)[..., None]
        np.save(
            os.path.join(
                args.dst_dir, f"{args.split}_fake_label_strings_with_dsetname.npy"
            ),
            fake_label_strings,
        )

    # divide by 1000 since depth is in mm
    transform = DepthToInverseDepth(scale_depth_before_inv=1 / 1000.0)
    mean_disparity, std_disparity, num_files = compute_mean_disparity(
        highres_depth_filenames, transform=transform, num_processes=args.num_processes
    )
    print(
        f"High res mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )

    # divide by 1000 since depth is in mm
    mean_disparity, std_disparity, num_files = compute_mean_disparity(
        lowres_depth_filenames, transform=transform, num_processes=args.num_processes
    )
    print(
        f"Low res mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )
    # compute range after normalization
    normalize_transform = torchvision.transforms.Compose(
        [
            transform,
            torchvision.transforms.Normalize(mean=mean_disparity, std=std_disparity),
        ]
    )
    mean_disparity, std_disparity, num_files = compute_mean_disparity(
        lowres_depth_filenames,
        transform=normalize_transform,
        num_processes=args.num_processes,
    )
    print(
        f"v1: Low res after norm. mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )

    normalize_transform = torchvision.transforms.Compose(
        [
            transform,
            DepthNorm(max_depth=None, compute_max_per_sample=True),
        ]
    )
    mean_disparity, std_disparity, num_files = compute_mean_disparity(
        lowres_depth_filenames,
        transform=normalize_transform,
        num_processes=args.num_processes,
    )
    print(
        f"v2: Low res after norm. mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )
    normalize_transform = torchvision.transforms.Compose(
        [
            transform,
            DepthNorm(max_depth=None, compute_max_per_sample=True),
            torchvision.transforms.Normalize(mean=mean_disparity, std=std_disparity),
        ]
    )
    mean_disparity, std_disparity, num_files = compute_mean_disparity(
        lowres_depth_filenames,
        transform=normalize_transform,
        num_processes=args.num_processes,
    )
    print(
        f"v3: Low res after norm. mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )


if __name__ == "__main__":
    main()
