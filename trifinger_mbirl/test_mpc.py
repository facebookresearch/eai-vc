import numpy as np
import argparse
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from trifinger_mbirl.ftpos_mpc import FTPosMPC
import utils.data_utils as d_utils

"""
Test forward rollout
Load demo trajectory with actions
Rollout from initial state with actions and compare to demo trajectory
"""

parser = argparse.ArgumentParser()
parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
args = parser.parse_args()

# Load demo.npz
data = np.load(args.file_path, allow_pickle=True)["data"]
traj = d_utils.get_traj_dict_from_obs_list(data)
#traj = d_utils.crop_traj_dict(d_utils.downsample_traj_dict(traj), [0, 30])
traj = d_utils.downsample_traj_dict(traj)

ft_pos_cur = traj["ft_pos_cur"]
delta_ftpos = traj["delta_ftpos"]

time_horizon = ft_pos_cur.shape[0]
x_init = ft_pos_cur[0, :]

# Run roll_out to get trajectory from initial state
ftpos_mpc = FTPosMPC(time_horizon)
ftpos_mpc.set_action_seq_for_testing(delta_ftpos)
x_traj = ftpos_mpc.roll_out(x_init)

x_traj = x_traj.detach().numpy()

# Compare against ground truth trajectory to check that rollout is correct
d_utils.plot_traj(
        "ft position", 
        None,
        ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
        {
        "pred":  {"y": x_traj, "x": traj["t"], "marker": "x"},
        "demo": {"y": ft_pos_cur, "x": traj["t"]},
        })
