#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import imageio
import torchvision.transforms as T
from PIL import Image

import trifinger_vc.control.finger_utils as f_utils


NON_TRAJ_KEYS = ["ft_pos_targets_per_mode"]


def get_traj_dict_from_obs_list(data, scale=1, include_image_obs=True):
    """
    Process list of observation dicts into dict of lists (trajectories for each quantity)

    args:
        data: list of observation dicts
        scale: amount to scale all distances by. by default, all distances are in meters. convert to cm with scale=100
    """
    data_keys = data[0].keys()

    if "object_observation" in data_keys:  # prev. key in obs
        position_error = np.array(
            [data[i]["achieved_goal"]["position_error"] for i in range(len(data))]
        )
        o_cur = np.array(
            [data[i]["object_observation"]["position"] for i in range(len(data))]
        )
        o_cur_ori = np.array(
            [data[i]["object_observation"]["orientation"] for i in range(len(data))]
        )
        robot_pos = np.array(
            [data[i]["robot_observation"]["position"] for i in range(len(data))]
        )
        o_des = np.array(
            [data[i]["desired_goal"]["position"] for i in range(len(data))]
        )
        o_des_ori = np.array(
            [data[i]["desired_goal"]["orientation"] for i in range(len(data))]
        )

    else:
        position_error = np.array(
            [data[i]["achieved_goal_position_error"] for i in range(len(data))]
        )
        o_cur = np.array([data[i]["object_position"] for i in range(len(data))])
        o_cur_ori = np.array([data[i]["object_orientation"] for i in range(len(data))])
        robot_pos = np.array([data[i]["robot_position"] for i in range(len(data))])
        o_des = np.array([data[i]["desired_goal"][:3] for i in range(len(data))])
        o_des_ori = np.array([data[i]["desired_goal"][3:] for i in range(len(data))])

    ft_pos_cur = np.array(
        [data[i]["policy"]["controller"]["ft_pos_cur"] for i in range(len(data))]
    )
    ft_pos_des = np.array(
        [data[i]["policy"]["controller"]["ft_pos_des"] for i in range(len(data))]
    )
    delta_ftpos = np.array([data[i]["action"]["delta_ftpos"] for i in range(len(data))])
    ft_vel_cur = np.array(
        [data[i]["policy"]["controller"]["ft_vel_cur"] for i in range(len(data))]
    )
    # ft_vel_des = np.array(
    #    [data[i]["policy"]["controller"]["ft_vel_des"] for i in range(len(data))]
    # )

    t = np.expand_dims(np.array([data[i]["t"] for i in range(len(data))]), 1)

    traj_dict = {
        "t": t,
        "o_pos_cur": scale * o_cur,
        "o_pos_des": scale * o_des,
        "o_ori_cur": o_cur_ori,
        "o_ori_des": o_des_ori,
        "ft_pos_cur": scale * ft_pos_cur,
        "ft_pos_des": scale * ft_pos_des,
        "ft_vel_cur": scale * ft_vel_cur,
        # "ft_vel_des": scale * ft_vel_des,
        "position_error": scale * position_error,
        "delta_ftpos": scale * delta_ftpos,
        "robot_pos": robot_pos,
    }

    if "scaled_success" in data_keys:
        scaled_success = np.array([data[i]["scaled_success"] for i in range(len(data))])
        traj_dict["scaled_success"] = scaled_success

    if include_image_obs:
        image60 = np.array(
            [
                data[i]["camera_observation"]["camera60"]["image"]
                for i in range(len(data))
            ]
        )
        image180 = np.array(
            [
                data[i]["camera_observation"]["camera180"]["image"]
                for i in range(len(data))
            ]
        )
        image300 = np.array(
            [
                data[i]["camera_observation"]["camera300"]["image"]
                for i in range(len(data))
            ]
        )
        traj_dict["image_60"] = image60
        traj_dict["image_180"] = image180
        traj_dict["image_300"] = image300

    # Mode information
    if "ft_pos_targets_per_mode" in data[-1]["policy"]:
        traj_dict["ft_pos_targets_per_mode"] = (
            scale * data[-1]["policy"]["ft_pos_targets_per_mode"]
        )

        # Add "mode"
        if "mode" not in data[0]["policy"]:
            traj_dict["mode"] = np.array(
                [
                    len(data[i]["policy"]["ft_pos_targets_per_mode"])
                    for i in range(len(data))
                ]
            )
        else:
            traj_dict["mode"] = np.array(
                [data[i]["policy"]["mode"] for i in range(len(data))]
            )

    # Object vertices
    if (
        "object_observation" in data[0] and "vertices" in data[0]["object_observation"]
    ) or ("object_vertices" in data[0]):
        vertices = []
        # Flatten vertices dict at each timestep and add to vertices list
        for i in range(len(data)):
            if "object_observation" in data[0]:
                v_wf_dict = data[i]["object_observation"]["vertices"]
            else:
                v_wf_dict = data[i]["object_vertices"]

            v_wf_flat = np.zeros(len(v_wf_dict) * 3)
            for k, v_wf in v_wf_dict.items():
                v_wf_flat[k * 3 : k * 3 + 3] = v_wf
            vertices.append(v_wf_flat)

        traj_dict["vertices"] = scale * np.array(vertices)
    else:
        # Vertices were not logged
        pass

    return traj_dict


def downsample_traj_dict(
    traj_dict,
    cur_time_step=0.004,
    new_time_step=0.1,
):
    """
    Downsample each of the trajectories in traj_dict.
    Add R3M embeddings for image_60

    args:
        traj_dict: dict of trajectories, generated by calling get_traj_dict_from_obs_list()
        cur_time_step: time step of raw observations (simulation/control timestep)
        new_time_step: time step to downsample to
    """

    every_x_steps = max(1, int(new_time_step / cur_time_step))
    num_waypoints = int(traj_dict["t"].shape[0] / every_x_steps)
    indices_to_take = (
        np.linspace(1, traj_dict["t"].shape[0], num_waypoints + 1, dtype=int) - 1
    )

    new_traj_dict = {}

    for k, traj in traj_dict.items():
        if "delta" in k:
            continue  # Need to recompute deltas for downsampled traj

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        # new_traj = traj[::every_x_steps, :]
        new_traj = traj[indices_to_take]
        new_traj_dict[k] = new_traj

    # Compute deltas for downsampled traj
    new_delta_ftpos = np.zeros(new_traj_dict["ft_pos_cur"].shape)
    ft_pos = new_traj_dict["ft_pos_des"]
    for t in range(ft_pos.shape[0] - 1):
        delta = ft_pos[t + 1] - ft_pos[t]
        new_delta_ftpos[t, :] = delta
    new_traj_dict["delta_ftpos"] = new_delta_ftpos

    return new_traj_dict


def get_traj_mode(traj_dict, mode):
    """Parse out part of trajectory with corresponding mode"""

    assert mode in [1, 2], "Invalid mode"

    new_traj_dict = {}

    indices_to_take = np.where(traj_dict["mode"] == mode)[0]
    for k, traj in traj_dict.items():
        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        new_traj = traj[indices_to_take]
        new_traj_dict[k] = new_traj

    return new_traj_dict


def crop_traj_dict(traj_dict, crop_range):
    """crop_range: [crop_min, crop_max]"""

    if crop_range[0] is None:
        crop_min = 0
    else:
        crop_min = crop_range[0]

    new_traj_dict = {}

    for k, traj in traj_dict.items():
        if crop_range[1] is None:
            crop_max = traj.shape[0]
        else:
            crop_max = crop_range[1]

        if k in NON_TRAJ_KEYS:
            new_traj_dict[k] = traj
            continue

        if traj.ndim == 2:
            new_traj = traj[crop_min:crop_max, :]
        else:
            new_traj = traj[crop_min:crop_max]
        new_traj_dict[k] = new_traj

    return new_traj_dict


def plot_traj(title, save_path, d_list, data_dicts, plot_timestamp=None):
    """
    Plot trajectories

    data_dicts = {
                 "label_1": {"y": y data, "x": x data (optional), "marker": marker string (optional)],},
                 "label_2": {"y": y data, "x"},
                 ...
                 }
    """

    plt.figure(figsize=(10, 10), dpi=200)
    plt.subplots_adjust(hspace=1)
    plt.suptitle(title)

    k = 0
    for i, d in enumerate(d_list):
        k += 1
        plt.subplot(len(d_list), 1, k)
        if len(d_list) > 1:
            plt.title(d)

        for label, data in data_dicts.items():
            num_steps = data["y"].shape[0]

            if "x" in data:
                x = data["x"]
            else:
                x = list(range(num_steps))

            if "marker" in data:
                marker = data["marker"]
            else:
                marker = None

            plt.plot(x, data["y"][:, i], marker=marker, label=label)

        if plot_timestamp is not None:
            plt.axvline(x=plot_timestamp, ls="--", c="k", lw=1)

    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def encode_img(model, transform, img):
    """
    Encode img by first passing it through transform, then through model
    ** Only works for single, unbatched image **
    """

    img_preproc = transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)

    return model(img_preproc)[0].detach()


def resize_img(img, new_dim=64):
    resize = T.Compose(
        [T.Resize(new_dim, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
    )
    resized_img = resize(Image.fromarray(img.astype(np.uint8)))
    # resized_img = resized_img.detach().numpy().transpose(1,2,0) * 255.0
    return resized_img


def save_gif(images, save_str, duration=None):
    frames = []
    for i, img in enumerate(images):
        # img = resize_img(img).detach().numpy().transpose(1,2,0) * 255.
        frames.append(img.astype(np.uint8))
    if duration is None:
        imageio.mimsave(save_str, frames)
    else:
        imageio.mimsave(save_str, frames, duration=duration)


def add_actions_to_obs(observation_list):
    """
    Given observation list with ft_pos_cur,
    add delta_ftpos and delta_q actions to observation.
    """

    for t in range(len(observation_list) - 1):
        ftpos_cur = observation_list[t]["policy"]["controller"]["ft_pos_cur"]
        ftpos_next = observation_list[t + 1]["policy"]["controller"]["ft_pos_cur"]
        delta_ftpos = ftpos_next - ftpos_cur

        q_cur = observation_list[t]["robot_position"]
        q_next = observation_list[t + 1]["robot_position"]
        delta_q = q_next - q_cur

        action_dict = {"delta_ftpos": delta_ftpos, "delta_q": delta_q}
        observation_list[t]["action"] = action_dict

    action_dict = {
        "delta_ftpos": np.zeros(delta_ftpos.shape),
        "delta_q": np.zeros(delta_q.shape),
    }
    observation_list[-1]["action"] = action_dict

def get_per_finger_ftpos_err(pred_ftpos, gt_ftpos, fnum=3):
    """Compute ftpos L2 distance for each finger"""

    ftpos_err = np.ones((pred_ftpos.shape[0], fnum)) * np.nan
    for i in range(fnum):
        per_finger_err = np.linalg.norm(
            (pred_ftpos[:, i * 3 : i * 3 + 3] - gt_ftpos[:, i * 3 : i * 3 + 3]),
            axis=1,
        )
        ftpos_err[:, i] = per_finger_err
    return ftpos_err


def get_reach_scaled_err(
    finger_to_move_list, init_ft_pos, cur_ft_pos, cube_pos, cube_half_size
):
    """Given list of finger ids to move, compute average scaled error"""

    total_scaled_err = 0
    for finger_to_move in finger_to_move_list:
        cur_ft_pos_i = cur_ft_pos[3 * finger_to_move : 3 * finger_to_move + 3]
        cur_dist_to_obj = max(
            np.linalg.norm(cur_ft_pos_i - cube_pos) - cube_half_size, 0
        )
        init_ft_pos_i = init_ft_pos[3 * finger_to_move : 3 * finger_to_move + 3]
        init_dist_to_obj = np.linalg.norm(init_ft_pos_i - cube_pos) - cube_half_size
        if init_dist_to_obj <= 0:
            # To prevent divide-by-0 error
            init_dist_to_obj = np.linalg.norm(init_ft_pos_i - cube_pos)
        scaled_err = min(1, (cur_dist_to_obj / init_dist_to_obj))
        total_scaled_err += scaled_err

    avg_scaled_err = total_scaled_err / len(finger_to_move_list)

    return avg_scaled_err