#!/usr/bin/env python3

import argparse
import functools
import glob
import os
import pickle
import tarfile
from multiprocessing import Pool

import numpy as np


# example command: python extra_scripts/preprocess_laion_subsets.py --root_dir /large_experiments/creativity/datasets/laion/laion400m-met-release/laion400m-dataset --dst_file /checkpoint/imisra/datasets/laion/laion400m_subset10m_tarlist.pkl --subset_length 10e6 --cache_file /checkpoint/imisra/datasets/laion/laion400m_numfiles.npy --full_data_output_file /checkpoint/imisra/datasets/laion/laion400m_tarlist.pkl
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    help="Root dir which contains all tars",
    default=None,
)
parser.add_argument("--root_dir_nested", default=False, action="store_true")
parser.add_argument("--full_data_input_file", type=str, default=None)
parser.add_argument(
    "--subset_length", type=float, required=True, help="Length of the subset"
)
parser.add_argument("--dst_file", type=str, required=True)
parser.add_argument("--cache_file", type=str, required=True)
parser.add_argument("--full_data_output_file", type=str, default=None)
parser.add_argument(
    "--image_extension", type=str, default="jpg", help="Length of the subset"
)
parser.add_argument(
    "--num_processes", type=int, default=70, help="num processes for mp.Pool"
)
parser.add_argument("--seed", type=int, default=0, help="seed for randomization")


def get_filenames_in_tar(fname, image_extension):
    with tarfile.open(fname) as tar_fh:
        tar_fnames = tar_fh.getnames()
    tar_fnames = [x for x in tar_fnames if x.endswith(image_extension)]
    return len(tar_fnames)


def main():
    args = parser.parse_args()
    if args.root_dir and not args.root_dir_nested:
        tarlist = sorted(glob.glob(args.root_dir + "/*tar"))
    elif args.root_dir and args.root_dir_nested:
        tarlist = sorted(glob.glob(args.root_dir + "/*/*tar"))
    if args.full_data_input_file:
        with open(args.full_data_input_file, "rb") as fh:
            tarlist = pickle.load(fh)
    tar_to_numfiles = []
    args.subset_length = int(args.subset_length)
    assert args.subset_length > 0
    num_tars = len(tarlist)
    print(f"Found {num_tars} tar files")

    if args.full_data_output_file:
        with open(args.full_data_output_file, "wb") as fh:
            pickle.dump(tarlist, fh)
        print(f"Wrote full dataset to {args.full_data_output_file}")

    if os.path.isfile(args.cache_file):
        print(f"Loading file number info from {args.cache_file}")
        tar_to_numfiles = np.load(args.cache_file)
    else:
        with Pool(args.num_processes) as p:
            tar_to_numfiles = p.map(
                functools.partial(
                    get_filenames_in_tar, image_extension=args.image_extension
                ),
                tarlist,
            )
        tar_to_numfiles = np.array(tar_to_numfiles, dtype=np.int64)
        np.save(args.cache_file, tar_to_numfiles)

    num_total_files = tar_to_numfiles.sum()
    print(f"Found total of {num_total_files} files in the dataset")
    assert (
        args.subset_length <= num_total_files
    ), f"Cannot make subset of {args.subset_length} from {num_total_files}"

    # shuffle tar names
    np.random.seed(args.seed)
    rnd_inds = np.random.permutation(num_tars)

    num_files_so_far = 0
    tarlist_subset = []
    tarlist_subset_count = []
    for ind in rnd_inds:
        tar_name = tarlist[ind]
        tarlist_subset.append(tar_name)
        num_files_so_far += tar_to_numfiles[ind]
        tarlist_subset_count.append(tar_to_numfiles[ind])
        if num_files_so_far >= args.subset_length:
            break

    print(
        f"Subset of {len(tarlist_subset)} tarfiles with {num_files_so_far} files of {args.image_extension} extension"
    )
    with open(args.dst_file, "wb") as fh:
        pickle.dump(tarlist_subset, fh)
    dst_count_file = args.dst_file.replace(".pkl", "_numfiles.npy")
    tarlist_subset_count = np.array(tarlist_subset_count, dtype=np.int64)
    np.save(dst_count_file, tarlist_subset_count)


if __name__ == "__main__":
    main()
