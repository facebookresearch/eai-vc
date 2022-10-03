import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torchvision
from omnivore.data.transforms.image_rgbd import DepthNorm
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--depth_names_file",
    type=str,
    required=True,
)
parser.add_argument(
    "--num_processes", type=int, default=10, help="num processes for mp.Pool"
)
parser.add_argument(
    "--fake_labels_dst_dir",
    type=str,
    default=None,
)
parser.add_argument(
    "--split",
    type=str,
    default=None,
)


def read_file_and_compute_stats(fname, transform):
    depth = torch.load(fname).float()
    depth = depth[..., None].permute(2, 0, 1)
    # add fake rgb channels
    fake_rgb = torch.zeros_like(depth)
    depth = torch.cat([fake_rgb, fake_rgb, fake_rgb, depth], dim=0)
    depth = transform(depth)
    depth = depth[3, ...]
    curr_mean = depth.mean().item()
    curr_std = depth.std().item()
    curr_max = depth.max().item()
    curr_min = depth.min().item()

    return [curr_mean, curr_std, curr_max, curr_min]


def compute_stats_helpers(depth_filenames, transform, num_processes):
    num_files = len(depth_filenames)

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

    print(f"Using transform {transform}")

    print(
        f"Mean disparity: {mean_disparity}, std: {std_disparity} computed over {num_files} samples"
    )
    print(f"Max overall {maxs}; Min overall {mins}")
    return mean_disparity, std_disparity


def compute_stats_main(args, depth_filenames):
    depth_norm_transform = DepthNorm(max_depth=75, clamp_max_before_scale=True)

    subset_inds = np.linspace(
        0, len(depth_filenames), num=1000, endpoint=False, dtype=np.int64
    )
    depth_filenames = [depth_filenames[x] for x in subset_inds]
    mean, std = compute_stats_helpers(
        depth_filenames, depth_norm_transform, args.num_processes
    )

    normalize_transform = torchvision.transforms.Compose(
        [
            depth_norm_transform,
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    mean, std = compute_stats_helpers(
        depth_filenames, normalize_transform, args.num_processes
    )

    normalize_transform = torchvision.transforms.Compose(
        [
            depth_norm_transform,
        ]
    )
    mean, std = compute_stats_helpers(
        depth_filenames, normalize_transform, args.num_processes
    )

    normalize_transform = torchvision.transforms.Compose(
        [
            DepthNorm(max_depth=None, compute_max_per_sample=True),
        ]
    )
    mean, std = compute_stats_helpers(
        depth_filenames, normalize_transform, args.num_processes
    )

    normalize_transform = torchvision.transforms.Compose(
        [
            DepthNorm(max_depth=None, compute_max_per_sample=True),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    compute_stats_helpers(depth_filenames, normalize_transform, args.num_processes)


def make_fake_labels(args, depth_filenames):
    assert args.split is not None
    fake_labels = np.arange(len(depth_filenames)).astype(np.int64)
    np.save(
        os.path.join(args.fake_labels_dst_dir, f"{args.split}_fake_labels.npy"),
        fake_labels,
    )
    fake_label_strings = [f"{x:09d}" for x in fake_labels]
    fake_label_strings = np.array(fake_label_strings)[..., None]
    np.save(
        os.path.join(args.fake_labels_dst_dir, f"{args.split}_fake_label_strings.npy"),
        fake_label_strings,
    )

    # Add dataset identifier
    fake_label_strings = [f"SUN RGB-D {x:09d}" for x in fake_labels]
    fake_label_strings = np.array(fake_label_strings)[..., None]
    np.save(
        os.path.join(
            args.fake_labels_dst_dir,
            f"{args.split}_fake_label_strings_with_dsetname.npy",
        ),
        fake_label_strings,
    )


def main():
    args = parser.parse_args()

    filenames = np.load(args.depth_names_file)
    depth_filenames = [os.path.join(args.root_dir, x) for x in filenames]

    if args.fake_labels_dst_dir:
        make_fake_labels(args, depth_filenames)
    else:
        compute_stats_main(args, depth_filenames)


if __name__ == "__main__":
    main()
