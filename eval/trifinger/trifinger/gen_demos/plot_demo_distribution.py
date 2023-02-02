import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

import utils.train_utils as t_utils


def main(args):
    # Load train and test trajectories
    with open(args.data_path, "r") as f:
        traj_info = json.load(f)
    train_traj_stats = traj_info["train_demo_stats"]
    test_traj_stats = traj_info["test_demo_stats"]

    # Get traj lists (read from demo files) and add to traj_info
    traj_info["train_demos"] = t_utils.get_traj_list(train_traj_stats, "pos")
    traj_info["test_demos"] = t_utils.get_traj_list(test_traj_stats, "pos")

    fig, axs = plt.subplots(2, 2, figsize=[10, 10])
    plt.subplots_adjust(hspace=0.5)

    # Iterate through demo-*.npz files and get init and final object positions
    # TODO maybe orientations also?
    r = -1
    for split in ["train", "test"]:
        init_pos_list = []
        goal_pos_list = []
        r += 1
        for demo in traj_info[f"{split}_demos"]:
            init_obj_pos = demo["o_pos_cur"][0, :]
            goal_obj_pos = demo["o_pos_cur"][-1, :]

            init_pos_list.append(init_obj_pos)
            goal_pos_list.append(goal_obj_pos)

        init_pos_arr = np.array(init_pos_list)
        goal_pos_arr = np.array(goal_pos_list)

        circle = plt.Circle((0, 0), 15, color="r", fill=False)
        axs[r, 0].title.set_text(f"{split} goal object positions")
        axs[r, 0].scatter(x=goal_pos_arr[:, 0], y=goal_pos_arr[:, 1])
        # change default range so that new circles will work
        axs[r, 0].set_xlim((-16, 16))
        axs[r, 0].set_ylim((-16, 16))
        axs[r, 0].set_aspect(1)
        axs[r, 0].add_patch(circle)

        # plt.title(f"{split} split initial (x,y) positions")
        circle = plt.Circle((0, 0), 15, color="r", fill=False)
        axs[r, 1].title.set_text(f"{split} initial object positions")
        axs[r, 1].scatter(x=init_pos_arr[:, 0], y=init_pos_arr[:, 1])
        # change default range so that new circles will work
        axs[r, 1].set_xlim((-16, 16))
        axs[r, 1].set_ylim((-16, 16))
        axs[r, 1].set_aspect(1)
        axs[r, 1].add_patch(circle)

    fig.suptitle(
        f"Init and goal object pos distribution for:\n{os.path.basename(args.data_path)}",
        fontsize=12,
    )
    save_path = f"{os.path.splitext(args.data_path)[0]}_distr.png"
    plt.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required for specifying training and test trajectories
    parser.add_argument(
        "data_path",
        help="Path to data.json file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
