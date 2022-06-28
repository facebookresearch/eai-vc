import sys
import os
import os.path
import argparse
import numpy as np
import time
from datetime import date, datetime
import multiprocessing


def main(args):
    
    log_paths = get_log_paths(args)


    log_paths_str = " ".join(log_paths)
    arg_str = "-d {DIFFICULTY} -l {LOG_PATHS}".format(DIFFICULTY = args.difficulty, LOG_PATHS = log_paths_str)
        
    if args.visualize: arg_str += " -v"

    cmd = f"python sim_move_cube.py {arg_str}"

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
    parser.add_argument("--difficulty", "-d", type=int, choices=[1,2,3], help="Difficulty level", default=1)
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument("--num_demos", "-nd", type=int, help="Number of demos to generate")
    parser.add_argument("--demo_dir", type=str, default="/Users/clairelchen/logs/demos/", help="Directory for demo logs")
    # TODO num proc
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
