import numpy as np
import argparse
import os
import sys
import json

import utils.data_utils as d_utils
import utils.train_utils as t_utils
from utils.preprocess_trajs import SCALE, get_dts_dir_name

"""
Load train and test trajectories and save them in a .json file. Need to already have run utils/preprocess_trajs.py to generate downsampled demos for your desired dts.

Example usage:
python prep_data/create_dataset_json.py --top_demo_dir /private/home/clairelchen/projects/demos_reach_colored_cube --dts 0.02 --diff_train 110 --diff_test 110 --r_train 0-100 --r_test 100-125

will generate the .json file: /private/home/clairelchen/projects/demos_reach_colored_cube/preloaded_dataset_stats/demos_dtrain-110_train-0-100_dtest-110_test-100-125_scale-100_dts-0p02.json

args:
    --top_demo_dir: path to demo_dir/
    --dts (float): downsample timestep, should be divisible by 0.004 (sim timestep)
    --diff_train: List of difficulties for training set
    --r_train: List of traj id ranges for each difficulty in diff_train list
        specified in format "min-max"
    --diff_test: List of difficulties for est set
    --r_test: List of traj id ranges for each difficulty in diff_test list
        specified in format "min-max"
"""


def save_traj_stats(traj_load_info, dts, save_path=None):
    """
    Load train and test trajectories from traj_load_info

    Args
        traj_load_info: a dict in the following format:
                        {
                            "demo_dir"   : top-level directory containing demos ("demos/"),
                            "difficulty" : difficulty level (1),
                            "train_demos": list of demo ids for training ([0,1]),
                            "test_demos" : list of demo ids for testing ([5]),
                        }
        save_path (str): If specified, save demo info in save_path
        scale: amount to scale distances by
        mode (int): 1 or 2; if specified, only return part of trajectory with this mode
    """

    def get_demo_dir(demo_dir, diff, dts, demo_id):
        dts_dir_name = get_dts_dir_name(dts)
        demo_path = os.path.join(
            demo_dir, f"difficulty-{diff}", f"demo-{demo_id:04d}", dts_dir_name
        )

        return demo_path

    top_demo_dir = traj_load_info["demo_dir"]

    train_demo_stats = []
    test_demo_stats = []

    # Load and downsample test trajectories for each difficulty
    for i, diff in enumerate(traj_load_info["diff_train"]):
        train_demo_ids = traj_load_info["train_demos"][i]
        for demo_id in train_demo_ids:
            demo_dir = get_demo_dir(top_demo_dir, diff, dts, demo_id)
            demo_stats = {"path": demo_dir, "diff": diff, "id": demo_id}
            train_demo_stats.append(demo_stats)

    for i, diff in enumerate(traj_load_info["diff_test"]):
        test_demo_ids = traj_load_info["test_demos"][i]
        for demo_id in test_demo_ids:
            demo_dir = get_demo_dir(top_demo_dir, diff, dts, demo_id)
            demo_stats = {"path": demo_dir, "diff": diff, "id": demo_id}
            test_demo_stats.append(demo_stats)

    print(f"Loaded {len(train_demo_stats)} training demos")
    print(f"Loaded {len(test_demo_stats)} test demos")

    train_traj_list = t_utils.get_traj_list(train_demo_stats, "pos")
    # Get max and min actions
    min_a, max_a = t_utils.get_traj_range(
        train_traj_list, "delta_ftpos", traj_stats=train_demo_stats
    )

    min_a_per_dim, max_a_per_dim = t_utils.get_traj_range_per_dim(
        train_traj_list, "delta_ftpos", traj_stats=train_demo_stats
    )

    max_a = np.ceil(max(abs(min_a), abs(max_a)))

    # Save object and finger type (read from demo_dir)
    # TODO hardcoded - perhaps there is a better way of doing this
    # if we collect more demos
    if "colored_cube" in traj_load_info["demo_dir"]:
        object_type = "colored_cube"
        finger_type = "trifinger_meta"
    elif "restruc" in traj_load_info["demo_dir"]:
        object_type = "colored_cube"
        finger_type = "trifingerpro"
    elif "green_cube" in traj_load_info["demo_dir"]:
        object_type = "green_cube"
        finger_type = "trifinger_meta"
    else:
        raise NameError

    # Save demo info (train and test demos)
    if save_path is not None:
        data = {
            "train_demo_stats": train_demo_stats,
            "test_demo_stats": test_demo_stats,
            "downsample_time_step": dts,
            "scale": SCALE,
            "max_a": max_a,
            "min_a_per_dim": min_a_per_dim.tolist(),
            "max_a_per_dim": max_a_per_dim.tolist(),
            "object_type": object_type,
            "finger_type": finger_type,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def main(args):

    save_dir = os.path.join(args.top_demo_dir, "preloaded_dataset_stats")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_demo_ids = []  # List of train demo ids for each difficulty
    test_demo_ids = []  # List of test demo ids for each difficulty

    diff_str_train = "_".join(map(str, args.diff_train))
    if len(args.diff_test) > 0:
        diff_str_test = "_".join(map(str, args.diff_test))
    else:
        diff_str_test = "none"

    r_str_train = "_".join(args.r_train)
    if args.r_test is not None:
        r_str_test = "_".join(args.r_test)
    else:
        r_str_test = "none"

    assert args.diff_train is not None

    for d_range_str in args.r_train:
        d_range = d_range_str.split("-")
        train_demo_ids.append(list(range(int(d_range[0]), int(d_range[1]))))

    if args.r_test is not None:
        for d_range_str in args.r_test:
            d_range = d_range_str.split("-")
            test_demo_ids.append(list(range(int(d_range[0]), int(d_range[1]))))

    dts_dir_name = get_dts_dir_name(args.dts)
    save_str = f"demos_dtrain-{diff_str_train}_train-{r_str_train}_dtest-{diff_str_test}_test-{r_str_test}_scale-{SCALE}_{dts_dir_name}.json"
    save_path = os.path.join(save_dir, save_str)

    # Info for loading trajectories
    traj_load_info = {
        "demo_dir": args.top_demo_dir,
        "diff_train": args.diff_train,
        "diff_test": args.diff_test,
        "train_demos": train_demo_ids,
        "test_demos": test_demo_ids,
    }

    save_traj_stats(traj_load_info, args.dts, save_path=save_path)


def parse_args():

    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument(
        "--top_demo_dir",
        default=f"/private/home/clairelchen/projects/demos_restruc/",
        help="Directory containing demos",
    )
    parser.add_argument(
        "--diff_train", type=int, nargs="*", default=[1], help="Difficulty levels"
    )
    parser.add_argument(
        "--diff_test", type=int, nargs="*", default=[], help="Difficulty levels"
    )

    parser.add_argument("--dts", type=float, default=0.2, help="Downsample time step")

    # Specify ranges of training and test trajectory ides
    parser.add_argument(
        "--r_train",
        type=str,
        nargs="*",
        default=None,
        help="IDs of training trajectories",
    )
    parser.add_argument(
        "--r_test", type=str, nargs="*", default=None, help="IDs of test trajectories"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
