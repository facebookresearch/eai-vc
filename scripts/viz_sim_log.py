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

def plot_traj(title, save_path, d_list, data_obs, data_des = None, plot_timestamp = None):
    num_steps = data_obs.shape[0]

    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplots_adjust(hspace=1)
    plt.suptitle(title)

    k = 0
    for i, d in enumerate(d_list):
        k += 1

        plt.subplot(len(d_list), 1, k)
        if len(d_list) > 1:
            plt.title(d)

        plt.plot(list(range(num_steps)), data_obs[:, i], marker=".", label="observed")

        if data_des is not None:
            plt.plot(list(range(num_steps)), data_des[:, i], label="desired")
            plt.legend()

    if plot_timestamp is not None:
        plt.axvline(x=plot_timestamp, ls='--', c='k', lw=1)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def main(file_path):
    data = np.load(file_path, allow_pickle=True)["data"]
    print(len(data))
    traj_dict = d_utils.get_traj_dict_from_obs_list(data)

    #position_error = np.array([data[i]["achieved_goal"]["position_error"] for i in range(len(data))])
    o_cur = traj_dict["o_cur"] # object position, observed
    o_des = traj_dict["o_des"] # object position, desired
    ft_pos_cur = traj_dict["ft_pos_cur"] # ft position, actual
    ft_pos_des = traj_dict["ft_pos_des"] # ft position, desired
    
    ## Plot ft positions
    plot_traj(
            "ft position", 
            None,
            ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3",],
            ft_pos_cur,
            ft_pos_des,
            )

    # Plot obj position trajectory
    plot_traj(
            "object position", 
            None,
            ["x", "y", "z"],
            o_cur,
            o_des,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    args = parser.parse_args()
    main(args.file_path)


