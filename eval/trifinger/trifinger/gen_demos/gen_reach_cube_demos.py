import sys
import os
import os.path
import argparse
import numpy as np
import time
from datetime import date, datetime
import multiprocessing

DIFFICULTY_CHOICES = [1, 2, 3]

# TODO could have difficulties 2 and 3 being reaching 2 or 3 fingers
# Rand init pos denoted by 1 in tens-place
# Rand init q denoted by 1 in ones-place
DIFF_CHOICES_RAND_INIT_POS = [110, 210, 310]
DIFF_CHOICES_RAND_INIT_Q = [101, 201, 301]
DIFF_CHOICES_RAND_INIT_POS_AND_Q = [111, 211, 311]
ALL_RAND = (
    DIFF_CHOICES_RAND_INIT_POS
    + DIFF_CHOICES_RAND_INIT_Q
    + DIFF_CHOICES_RAND_INIT_POS_AND_Q
)


ALL_MODES = DIFFICULTY_CHOICES + ALL_RAND


def main(args):
    for mode in args.modes:
        if mode not in ALL_MODES:
            print(f"Mode {mode} not valid. Skipping")
            continue

        print(f"Generating demos for mode {mode}")

        log_paths = get_log_paths(args, mode)

        if mode in ALL_RAND:
            diff = int(mode / 100)
        else:
            diff = mode

        log_paths_str = " ".join(log_paths)
        arg_str = "-d {DIFFICULTY} -l {LOG_PATHS}".format(
            DIFFICULTY=diff, LOG_PATHS=log_paths_str
        )

        if args.visualize:
            arg_str += " -v"

        if mode in DIFFICULTY_CHOICES:
            cmd = f"python sim_reach_cube.py {arg_str}"
        elif mode in DIFF_CHOICES_RAND_INIT_POS:
            cmd = f"python sim_reach_cube.py {arg_str} -rp"
        elif mode in DIFF_CHOICES_RAND_INIT_Q:
            cmd = f"python sim_reach_cube.py {arg_str} -rq"
        elif mode in DIFF_CHOICES_RAND_INIT_POS_AND_Q:
            cmd = f"python sim_reach_cube.py {arg_str} -rp -rq"
        else:
            raise ValueError("Invalid difficulty.")

        # Run command
        os.system(cmd)


def get_log_paths(args, mode):
    log_dir = os.path.join(args.demo_dir, f"difficulty-{mode}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path_list = []
    for i in range(args.num_demos):
        demo_name = f"demo-{i:04d}"
        demo_dir = os.path.join(log_dir, demo_name)
        if not os.path.exists(demo_dir):
            os.makedirs(demo_dir)
        demo_file_name = f"{demo_name}.npz"
        log_path = os.path.join(demo_dir, demo_file_name)
        log_path_list.append(log_path)

    return log_path_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        "-m",
        nargs="*",
        type=int,
        help="Difficulty level / mode",
        default=[1],
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument(
        "--num_demos", "-nd", type=int, help="Number of demos to generate"
    )
    parser.add_argument(
        "--demo_dir", type=str, default="tmp/demos/", help="Directory for demo logs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
