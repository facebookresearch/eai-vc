import numpy as np
import argparse
import os
import sys
import json

import utils.data_utils as d_utils
import utils.train_utils as t_utils
from utils.preprocess_trajs import SCALE, get_dts_dir_name

"""
Load train and test trajectories and save them in a .pth file
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
    for i, diff in enumerate(traj_load_info["difficulty"]):
        train_demo_ids = traj_load_info["train_demos"][i]
        test_demo_ids = traj_load_info["test_demos"][i]

        for demo_id_list, stats_list in [
            [train_demo_ids, train_demo_stats],
            [test_demo_ids, test_demo_stats],
        ]:

            for demo_id in demo_id_list:
                demo_dir = get_demo_dir(top_demo_dir, diff, dts, demo_id)

                demo_stats = {"path": demo_dir, "diff": diff, "id": demo_id}
                stats_list.append(demo_stats)

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

    diff_str = "_".join(map(str, args.difficulty))

    if args.id_train is None:
        for i in range(len(args.difficulty)):
            n_train = args.n_train[i]
            n_test = args.n_test[i]

            train_demo_ids.append(list(range(n_train)))
            test_demo_ids.append(list(range(n_train, n_train + n_test)))

        n_train_str = "_".join(map(str, args.n_train))
        n_test_str = "_".join(map(str, args.n_test))
        dts_dir_name = get_dts_dir_name(args.dts)
        save_str = f"demos_d-{diff_str}_train-{n_train_str}_test-{n_test_str}_scale-{SCALE}_{dts_dir_name}.json"
        save_path = os.path.join(save_dir, save_str)

    else:
        train_demo_ids.append(args.id_train)

        if args.id_test:
            test_demo_ids.append(args.id_test)
            id_test_str = "_".join(map(str, args.id_test))
        else:
            test_demo_ids.append([])
            id_test_str = "none"

        id_train_str = "_".join(map(str, args.id_train))
        dts_dir_name = get_dts_dir_name(args.dts)
        save_str = f"demos_d-{diff_str}_train_id-{id_train_str}_test_id-{id_test_str}_scale-{SCALE}_{dts_dir_name}.json"
        save_path = os.path.join(save_dir, save_str)

    # Info for loading trajectories
    traj_load_info = {
        "demo_dir": args.top_demo_dir,
        "difficulty": args.difficulty,
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
        "--difficulty", type=int, nargs="*", default=[1], help="Difficulty levels"
    )

    parser.add_argument("--dts", type=float, default=0.2, help="Downsample time step")

    # Two different ways of specifying training and test trajectory splits
    # Method 1: Specify number of training and test trajectories for each difficulty
    # with --n_train and --n_test
    parser.add_argument(
        "--n_train",
        type=int,
        nargs="*",
        default=[10],
        help="Number of training trajectories",
    )
    parser.add_argument(
        "--n_test", type=int, nargs="*", default=[0], help="Number of test trajectories"
    )
    # Method 2: Specify IDs of training and test trajectories with --id_train and --id_test
    # TODO For now, will only work with one difficulty specified
    parser.add_argument(
        "--id_train",
        type=int,
        nargs="*",
        default=None,
        help="IDs of training trajectories",
    )
    parser.add_argument(
        "--id_test", type=int, nargs="*", default=None, help="IDs of test trajectories"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
