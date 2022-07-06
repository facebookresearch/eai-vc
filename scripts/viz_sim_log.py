import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import os.path
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

def main(args):
    file_path = args.file_path

    data = np.load(file_path, allow_pickle=True)["data"]
    print(len(data))
    traj_dict = d_utils.get_traj_dict_from_obs_list(data)

    #position_error = np.array([data[i]["achieved_goal"]["position_error"] for i in range(len(data))])
    o_cur = traj_dict["o_cur_pos"] # object position, observed
    o_des = traj_dict["o_des_pos"] # object position, desired
    ft_pos_cur = traj_dict["ft_pos_cur"] # ft position, actual
    ft_pos_des = traj_dict["ft_pos_des"] # ft position, desired

    o_cur_ori = traj_dict["o_cur_ori"] # object ori observed
    o_des_ori = traj_dict["o_des_ori"] # object ori desired

    downsampled_traj_dict = d_utils.downsample_traj_dict(traj_dict)
    
    
    if args.save:
        demo_name = os.path.splitext(os.path.split(file_path)[1])[0]
        demo_dir = os.path.split(file_path)[0]
        out_dir = os.path.join(demo_dir, "sim_viz", demo_name)
        if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=False)
    else:
        out_dir = None

    ## Plot ft positions
    d_utils.plot_traj(
            "ft position", 
            (os.path.join(out_dir, "ft_pos.png") if out_dir else None),
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            {
            "desired":  {"y": traj_dict["ft_pos_des"], "x": traj_dict["t"]},
            "actual":  {"y": traj_dict["ft_pos_cur"], "x": traj_dict["t"]},
            #"down": {"y": downsampled_traj_dict["ft_pos_cur"], "x": downsampled_traj_dict["t"], "marker": "x"},
            }
            )

    ## Plot object positions
    d_utils.plot_traj(
            "object position", 
            (os.path.join(out_dir, "obj_pos.png") if out_dir else None),
            ["x", "y", "z"],
            {
            "actual":  {"y": o_cur, "x": traj_dict["t"]},
            "desired": {"y": o_des, "x": traj_dict["t"]},
            }
            )

    ## Plot object positions
    d_utils.plot_traj(
            "object ori", 
            (os.path.join(out_dir, "obj_ori.png") if out_dir else None),
            ["x", "y", "z", "w"],
            {
            "actual":  {"y": o_cur_ori, "x": traj_dict["t"]},
            }
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--save", "-s", action="store_true", help="Save figs")
    args = parser.parse_args()
    main(args)


