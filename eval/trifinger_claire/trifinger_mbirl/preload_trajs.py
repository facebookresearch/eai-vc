import torch
import numpy as np
import argparse
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, ".."))
sys.path.insert(0, os.path.join(base_path, "../.."))

import utils.data_utils as d_utils

"""
Load train and test trajectories and save them in a .pth file
"""


def main(args):

    save_dir = os.path.join(args.demo_dir, "preloaded_demos")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    diff_str = "_".join(map(str, args.difficulty))
    n_train_str = "_".join(map(str, args.n_train))
    n_test_str = "_".join(map(str, args.n_test))
    save_str = f"demos_d-{diff_str}_train-{n_train_str}_test-{n_test_str}_scale-{args.scale}.pth"
    save_path = os.path.join(save_dir, save_str)

    train_demo_ids = []  # List of train demo ids for each difficulty
    test_demo_ids = []  # List of test demo ids for each difficulty

    for i in range(len(args.difficulty)):
        n_train = args.n_train[i]
        n_test = args.n_test[i]

        train_demo_ids.append(list(range(n_train)))
        test_demo_ids.append(list(range(n_train, n_train + n_test)))

    # Info for loading trajectories
    traj_load_info = {
        "demo_dir": args.demo_dir,
        "difficulty": args.difficulty,
        "train_demos": train_demo_ids,
        "test_demos": test_demo_ids,
    }

    train_trajs, test_trajs = d_utils.load_trajs(
        traj_load_info, save_path=save_path, scale=args.scale
    )


def parse_args():

    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument(
        "--demo_dir",
        default=f"/private/home/clairelchen/projects/demos/",
        help="Directory containing demos",
    )
    parser.add_argument(
        "--difficulty", type=int, nargs="*", default=[1], help="Difficulty levels"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        nargs="*",
        default=[10],
        help="Number of training trajectories",
    )
    parser.add_argument(
        "--n_test", type=int, nargs="*", default=[1], help="Number of test trajectories"
    )
    parser.add_argument(
        "--scale", type=float, default=100, help="Amount to scale trajectories by"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
