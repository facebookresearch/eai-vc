import torch
import numpy as np
import argparse
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))
sys.path.insert(0, os.path.join(base_path, '../..'))

import utils.data_utils as d_utils

"""
Load train and test trajectories and save them in .json file
"""

def main(args):
    data_dir = os.path.dirname(args.file_path)
    train_trajs, test_trajs = d_utils.load_trajs(args.file_path, exp_dir=data_dir, scale=100)

def parse_args():

    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

