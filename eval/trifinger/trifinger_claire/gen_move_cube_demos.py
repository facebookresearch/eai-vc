import sys
import os
import os.path
import argparse
import numpy as np
import time
from datetime import date, datetime
import multiprocessing

DIFFICULTY_CHOICES = [0, 1, 2, 3]
RANDOM_ACTION_IDS = [8, 9]


def main(args):

    if args.difficulty in RANDOM_ACTION_IDS:
        assert (
            args.seed_demos_path is not None
        ), "Need to specify --seed_demos_path to preloaded_demos.pth"

    log_paths = get_log_paths(args)

    log_paths_str = " ".join(log_paths)
    arg_str = "-d {DIFFICULTY} -l {LOG_PATHS}".format(
        DIFFICULTY=args.difficulty, LOG_PATHS=log_paths_str
    )

    if args.visualize:
        arg_str += " -v"

    if args.difficulty in [0, 1, 2, 3]:
        cmd = f"python sim_move_cube.py {arg_str}"
    else:
        arg_str += f" -s {args.seed_demos_path}"
        cmd = f"python sim_random_actions.py {arg_str}"

    # Run sim_move_cube.py
    os.system(cmd)


def get_log_paths(args):

    log_dir = os.path.join(args.demo_dir, f"difficulty-{args.difficulty}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path_list = []
    for i in range(args.num_demos):
        demo_name = f"demo-{i:04d}.npz"
        log_path = os.path.join(log_dir, demo_name)
        log_path_list.append(log_path)

    return log_path_list


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        "-d",
        type=int,
        choices=DIFFICULTY_CHOICES + RANDOM_ACTION_IDS,
        help="Difficulty level",
        default=1,
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument(
        "--num_demos", "-nd", type=int, help="Number of demos to generate"
    )
    parser.add_argument(
        "--demo_dir", type=str, default="test/demos/", help="Directory for demo logs"
    )
    parser.add_argument(
        "--seed_demos_path", "-s", type=str, help="Path to preloaded_demos.pth"
    )
    # TODO num proc
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
